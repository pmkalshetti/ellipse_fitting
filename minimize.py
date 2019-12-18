import tensorflow as tf
from tensorflow_graphics.math.optimizer import levenberg_marquardt
import matplotlib.pyplot as plt
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
        distance = tf.norm(self.data.samples - points_on_model, axis=0)

        return tf.reduce_sum(distance)

    def minimize(self):
        value_obj, variables = levenberg_marquardt.minimize(
            residuals=self.objective,
            variables=[self.model.theta, self.parameters],
            max_iterations=100,
            callback=self.per_iteration_callback
        )

    def per_iteration_callback(self, idx_iter, val_obj, variables):
        print(f"Iteration: {idx_iter:03d}, Objective: {val_obj:.2f}")
        self.model.set_theta(variables[0])  # VERY IMPORTANT!
        self.plot()
        plt.pause(1)

    def plot(self):
        self.ax.clear()
        self.model.plot(self.ax)
        self.data.plot(self.ax)


if __name__ == "__main__":

    tf.random.set_seed(1)

    data = Data()
    model = Ellipse(tf.constant([1, 2, 3, 2, 1, 2.]))
    parameters_init = data.parameters \
        + tf.random.normal(data.parameters.shape, stddev=0.5)

    minimizer = Minimizer(data, model, parameters_init, True)
    minimizer.minimize()
