import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
import argparse
from data import Data
from model import Ellipse


class Minimizer:
    def __init__(
            self,
            data,
            model,
            parameters_init,
            flag_plot=False
    ):
        self.parameters = parameters_init
        self.model = model
        self.data = data
        self.flag_plot = flag_plot

        if self.flag_plot:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)

    def objective(self, theta, parameters):
        points_on_model = self.model.evaluate(parameters)

        # WEIRD! in below block norm axis is wrong, still works better!
        # distance = tf.norm(self.data.samples - points_on_model, axis=0)
        # return tf.reduce_sum(distance)

        distance = tf.reduce_sum(
            tf.norm(self.data.samples - points_on_model, axis=1)**2
        )
        return distance

    def minimize_tf_lm(self, flag_callback=False, n_iters=20):
        import tensorflow_graphics as tfg  # used here since import is slow
        value_obj, variables = tfg.math.optimizer.levenberg_marquardt.minimize(
            residuals=self.objective,
            variables=[self.model.theta, self.parameters],
            max_iterations=n_iters,
            callback=self.per_iteration_callback if flag_callback else None
        )

        return variables[0]

    def per_iteration_callback(self, idx_iter, val_obj, variables):
        print(f"Iteration: {idx_iter:2d}, Objective: {val_obj:.2f}")

        # Note: ellipse theta need to be updated manually.
        self.model.set_theta(variables[0])  # VERY IMPORTANT!
        self.plot()
        plt.pause(1)
        self.fig.suptitle(f"Iteration: {idx_iter:3d}")

        # save fig
        # self.fig.savefig(f"media_readme/{idx_iter:02d}.png")

    def plot(self):
        self.ax.clear()
        self.model.plot(self.ax)
        self.data.plot(self.ax)
        plt.legend()

    def minimize_my_lm(self, flag_callback=False, n_iters=20):
        """Manually implemented Levenberg Marquardt algorithm.

        References
        ----------
        https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/optimizer/levenberg_marquardt.py # noqa
        """

        residuals = [self.objective]
        variables = [self.model.theta, self.parameters]
        regularizer = 1e-20
        multiplier = 10.

        val_obj = tf.reduce_sum([
            tf.nn.l2_loss(residual(*variables))
            for residual in residuals
        ])
        for idx_iter in range(n_iters):
            values, jacobian = _values_and_jacobian(residuals, variables)

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
                val_obj_new = tf.reduce_sum([
                    tf.nn.l2_loss(residual(*variables_new))
                    for residual in residuals
                ])
                # if new estimated solution does not decrease the objective,
                # no updates are performed, but a new regularizer is computed
                if val_obj_new < val_obj:
                    val_obj = val_obj_new
                    variables = variables_new
                else:
                    regularizer *= multiplier

            except tf.errors.InvalidArgumentError:
                regularizer *= multiplier

            if flag_callback:
                self.per_iteration_callback(idx_iter+1, val_obj, variables)

        return variables[0]


def _values_and_jacobian(residuals, variables):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-tf", "--flag_tf", action="store_true",
        help="Use tensorflow implementation of Levenberg Marquardt"
    )
    parser.add_argument(
        "-c", "--flag_callback", action="store_true",
        help="Enable per iteration callback"
    )
    args = parser.parse_args()

    tf.random.set_seed(1)

    data = Data()
    model = Ellipse(tf.constant([1, 2, 3, 2, 1, 2.]))
    parameters_init = data.parameters \
        + tf.random.normal(data.parameters.shape, stddev=0.5)

    minimizer = Minimizer(data, model, parameters_init, True)

    if args.flag_tf:
        theta = minimizer.minimize_tf_lm(args.flag_callback)
    else:
        theta = minimizer.minimize_my_lm(args.flag_callback)

    print("Minimization Complete!")
    print(f"Ellipse parameters (theta): {theta}")
