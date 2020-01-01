import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import itertools
import argparse
from ellipse import get_points_on_ellipse


def generate_data_from_ellipse(
        params_ellipse,
        n_data=30, fraction=0.3, stddev_noise=0.08
):
    """Generates noisy points from a fraction on ellipse.

    Arguments
    ---------
    params_ellipse: tf.Tensor; (6,); tf.float32
        ellipse equation parameters.
    n_data: int
        number of data points to generate
    fraction: float
        denotes the fraction of the ellipse to sample points from.
        lies in [0, 1]
    stddev_noise: float
        standard deviation of Gaussian noise added to sampled points.

    Returns
    -------
    data: tf.Tensor; (`n_data`, 2); tf.float32
        noisy points on ellipse.
    """
    params_points = tf.linspace(0., 2*np.pi*fraction, n_data)
    data = get_points_on_ellipse(params_ellipse, params_points) \
        + tf.random.normal((n_data, 2), stddev=stddev_noise)

    return data


def compute_objective(residuals, variables):
    """Returns sum of squared of residuals."""
    return tf.reduce_sum([
            tf.nn.l2_loss(residual(*variables))
            for residual in residuals
        ])


def get_values_and_jacobian(residuals, variables):
    def _compute_residual_values(residiuals, variables):
        return [
            tf.unstack(tf.reshape(residual(*variables), [-1]))
            for residual in residuals
        ]

    def _compute_jacobian(values, variables, tape):
        jacobian = []
        for value in itertools.chain.from_iterable(values):
            gradient = tape.gradient(value, variables)
            gradient = [
                tf.zeros_like(v) if g is None else g
                for v, g in zip(variables, gradient)
            ]
            gradient = [tf.reshape(g, [-1]) for g in gradient]
            gradient = tf.concat(gradient, axis=0)
            jacobian.append(gradient)

        return tf.stack(jacobian)

    with tf.GradientTape(
            watch_accessed_variables=False, persistent=True
    ) as tape:
        for variable in variables:
            tape.watch(variable)
        values = _compute_residual_values(residuals, variables)
    jacobian = _compute_jacobian(values, variables, tape)
    del tape
    values = tf.expand_dims(tf.concat(values, axis=0), axis=-1)
    return values, jacobian


class Data:
    def __init__(self, params_ellipse,
                 n_data=30, fraction=0.3, stddev_noise=0.08):
        self.data = generate_data_from_ellipse(
            params_ellipse, n_data, fraction, stddev_noise
        )

    def fit_ellipse(self, params_ellipse_init, params_points_init,
                    n_iter=100, regularizer=1e-20, multiplier_regularizer=10.,
                    metric="absolute"):
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
        residuals = [
            lambda arg1, arg2: self.compute_data_term(arg1, arg2, metric)
        ]
        variables = [params_ellipse_init, params_points_init]
        value_objective = compute_objective(residuals, variables)

        # create canvas
        fig, ax = plt.subplots()
        ax.scatter(
            self.data[:, 0], self.data[:, 1], label="data",
            c="black", s=10
        )
        points_on_ellipse = get_points_on_ellipse(
            params_ellipse_init, tf.linspace(0., 2*np.pi, 100))
        plot_ellipse = ax.plot(
            points_on_ellipse[:, 0], points_on_ellipse[:, 1],
            label="model", c="orange",
        )
        plt.pause(1)

        # Levenberg Marquardt Algorithm
        for idx_iter in range(n_iter):
            values, jacobian = get_values_and_jacobian(residuals, variables)

            # solve normal equation
            try:
                updates = tf.linalg.lstsq(
                    jacobian, values, l2_regularizer=regularizer
                )
                shapes = [tf.shape(variable) for variable in variables]
                splits = [tf.reduce_prod(shape) for shape in shapes]
                updates = tf.split(tf.squeeze(updates, axis=-1), splits)
                variables_new = [
                    v - tf.reshape(u, s)
                    for v, u, s in zip(variables, updates, shapes)
                ]
                value_objective_new = compute_objective(
                    residuals, variables_new
                )
                # if new estimated solution does not decrease the objective,
                # no updates are performed, but a new regularizer is computed
                if value_objective_new < value_objective:
                    value_objective = value_objective_new
                    variables = variables_new
                else:
                    regularizer *= multiplier_regularizer

            except tf.errors.InvalidArgumentError:
                regularizer *= multiplier_regularizer

            print(
                f"Iteration: {idx_iter:3d}, Objective: {value_objective:.2f}"
            )

            # plot
            fig.suptitle(
                f"Iteration: {idx_iter:3d}, Objective: {value_objective:.3f}"
            )
            points_on_ellipse = get_points_on_ellipse(
                variables[0], tf.linspace(0., 2*np.pi, 100))
            plot_ellipse[0].set_data(
                points_on_ellipse[:, 0], points_on_ellipse[:, 1]
            )
            fig.canvas.draw()
            # fig.savefig(f"log/{metric}/{idx_iter:03d}.png")
        print("Fitting complete.")

        return variables[0]

    def compute_data_term(self, params_ellipse, params_points, metric):
        """Returns distance between ellipse and data.
        Arguments
        ---------
        params_ellipse: tf.Tensor; (6,); tf.float32
            ellipse equation parameters.
        params_points: tf.Tensor; (n_points,); tf.float32
            curve parameter which lies in [0, 2*np.pi].
        metric: string
            either "absolute" or "squared"
        """
        if metric not in ["absolute", "squared"]:
            raise ValueError(
                "`metric` must be either 'absolute' or 'squared'."
            )

        points_on_ellipse = get_points_on_ellipse(
            params_ellipse, params_points
        )

        if metric == "absolute":
            distance = tf.abs(self.data - points_on_ellipse)
        elif metric == "squared":
            distance = (self.data - points_on_ellipse) ** 2

        return tf.reduce_mean(distance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--metric", default="absolute",
        help="specify which metric to use from {absolute, metric}."
    )
    parser.add_argument(
        "-s", "--stddev_noise", default=0.1, type=float,
        help="standard deviation of Gaussian noise on data."
    )
    args = parser.parse_args()

    # generate data
    params_ellipse_true = tf.constant([3, 2, 5, 4, 5, 7.])
    data = Data(params_ellipse_true, stddev_noise=args.stddev_noise)

    # fit ellipse to data
    params_ellipse_init = tf.constant([1, 2, 3, 2, 1, 2.])
    fraction_of_ellipse = 0.5
    n_points_init = 30
    params_points_init = tf.linspace(
        0., 2*np.pi*fraction_of_ellipse, n_points_init
    )
    params_ellipse_final = data.fit_ellipse(
        params_ellipse_init, params_points_init, metric=args.metric
    )

    np.set_printoptions(precision=2, suppress=True)
    print((
        "Ellipse Parameters: \n"
        f"True: {params_ellipse_true}\n"
        f"Fitted: {params_ellipse_final}"
    ))
