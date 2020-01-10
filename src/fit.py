import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ellipse import get_points_on_ellipse
from data import generate_data_from_ellipse


class EllipseFitter:
    def __init__(self, params_ellipse=[3, 2, 5, 4, 5, 7.],
                 n_data=30, fraction=0.3, stddev_noise=0.08):
        self.data = generate_data_from_ellipse(
            params_ellipse, n_data, fraction, stddev_noise
        )

    @staticmethod
    def compute_jacobian(values, variables, tape):
        jacobian = []
        for value in values:
            gradient = tape.gradient(value, variables)
            gradient = [
                tf.zeros_like(v) if g is None else g
                for v, g in zip(variables, gradient)
            ]
            gradient = [tf.reshape(g, [-1]) for g in gradient]
            gradient = tf.concat(gradient, axis=0)
            jacobian.append(gradient)

        return tf.stack(jacobian)

    def evaluate_residuals(self, variables, metric="squared"):
        """Returns average squared/absolute difference."""
        points_on_ellipse = get_points_on_ellipse(*variables)

        if metric == "absolute":
            distance = tf.abs(self.data - points_on_ellipse)
        elif metric == "squared":
            distance = (self.data - points_on_ellipse) ** 2

        distance = tf.reduce_mean(
            tf.reduce_sum(
                distance,  # shape=(n,2)
                axis=-1  # sum along x,y
            ),  # shape=(n)
            axis=0  # mean along all points
        )

        # output should be list of residuals
        value_residuals = [
            distance
        ]

        return value_residuals

    def compute_error(self, variables):
        """Returns average euclidean norm of difference."""
        points_on_ellipse = get_points_on_ellipse(*variables)

        distance = tf.reduce_mean(
            tf.norm(
                self.data - points_on_ellipse,  # shape=(n,2)
                axis=-1  # norm along x,y
            ),  # shape=(n)
            axis=0  # mean along all points
        )

        return distance

    def fit(self, params_ellipse_init, params_points_init,
            n_iter=50, regularizer=1e-20, multiplier_regularizer=10.,
            flag_plot=False, flag_save=False):
        """Returns ellipse parameters that fit to the data.

        Arguments
        ---------
        params_ellipse_init: tf.Tensor; (6,); tf.float32
            ellipse equation parameters.
            initial parameters for optimization.
        params_points_init: tf.Tensor; (n_points,); tf.float32
            curve parameter which lies in [0, 2*np.pi].
            initial parameters for optimization.
        n_iter: int
            Number of optimizer iterations.
        regularizer: float
            Regularizer used in least-squares.
        multiplier_regularizer: float
            Update regularizer by this factor when objective is not reduced.

        Returns
        -------
        params_ellipse: tf.Tensor; (6,); tf.float32
            ellipse equation parameters.
            final parameters after optimization.

        Reference
        ---------
        The implementation was inspired from tensorflow_graphics
        https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/optimizer/levenberg_marquardt.py # noqa
        """
        params_ellipse = tf.convert_to_tensor(params_ellipse_init)
        params_points = tf.convert_to_tensor(params_points_init)
        variables = [params_ellipse, params_points]

        shapes = [tf.shape(variable) for variable in variables]
        splits = [tf.reduce_prod(shape) for shape in shapes]

        value_objective = tf.nn.l2_loss(
            self.evaluate_residuals(variables)
        )

        if flag_plot:
            # create canvas
            fig, ax = plt.subplots()
            ax.scatter(
                self.data[:, 0], self.data[:, 1], label="data",
                facecolors="none", edgecolors="black", s=20
            )
            points_on_ellipse = get_points_on_ellipse(
                params_ellipse, tf.linspace(0., 2*np.pi, 100))
            plot_ellipse = ax.plot(
                points_on_ellipse[:, 0], points_on_ellipse[:, 1],
                label="model", c="orange",
            )
            plt.pause(1)

            if flag_save:
                tf.io.gfile.makedirs("log_fit")

        # Levenberg Marquardt Algorithm
        for idx_iter in range(n_iter):
            # compute objective and gradients
            with tf.GradientTape(
                    watch_accessed_variables=False, persistent=True) as tape:
                for variable in variables:
                    tape.watch(variable)

                residuals = self.evaluate_residuals(variables)

            jacobian = self.compute_jacobian(
                residuals, variables, tape
            )
            del tape
            values = tf.stack(residuals, axis=0)[:, tf.newaxis]

            # solve normal equation
            try:
                updates = tf.linalg.lstsq(
                    jacobian, values, l2_regularizer=regularizer
                )
                updates = tf.split(tf.squeeze(updates, axis=-1), splits)

                # update
                variables_new = [
                    v - tf.reshape(u, s)
                    for v, u, s in zip(variables, updates, shapes)
                ]

                # compute new objective
                value_objective_new = tf.nn.l2_loss(
                    self.evaluate_residuals(variables_new)
                )

                # keep update if new objective < curr objective
                if value_objective_new < value_objective:
                    value_objective = value_objective_new
                    variables = variables_new
                else:
                    regularizer *= multiplier_regularizer
            except tf.errors.InvalidArgumentError:
                regularizer *= multiplier_regularizer

            error = self.compute_error(variables)
            str_log = (
                f"Iteration: {idx_iter:3d}, "
                f"Objective: {value_objective:.2f}, "
                f"Average Error: {error:.2f}"
            )
            print(str_log)

            if flag_plot:
                fig.suptitle(str_log)
                points_on_ellipse = get_points_on_ellipse(
                    variables[0], tf.linspace(0., 2*np.pi, 100))
                plot_ellipse[0].set_data(
                    points_on_ellipse[:, 0], points_on_ellipse[:, 1]
                )
                fig.canvas.draw()

                if flag_save:
                    fig.savefig(
                        f"log_fit/{idx_iter:03d}.png", bbox_inches="tight"
                    )
        print(f"{n_iter} fitting iterations completed.")
        return variables[0]


if __name__ == "__main__":
    tf.random.set_seed(1)

    # generate data
    params_ellipse_true = tf.constant([3, 2, 5, 4, 5, 7.])
    ellipse_fitter = EllipseFitter(params_ellipse_true)

    # fit ellipse to data
    params_ellipse_init = tf.constant([1, 2, 3, 2, 1, 2.])
    fraction_of_ellipse = 0.5
    n_points_init = 30
    params_points_init = tf.linspace(
        0., 2*np.pi*fraction_of_ellipse, n_points_init
    )

    params_ellipse_final = ellipse_fitter.fit(
        params_ellipse_init, params_points_init,
        flag_plot=True, flag_save=True
    )

    np.set_printoptions(precision=2, suppress=True)
    print((
        "Ellipse Parameters: \n"
        f"True: {params_ellipse_true}\n"
        f"Fitted: {params_ellipse_final}"
    ))
