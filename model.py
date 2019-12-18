import tensorflow as tf
import numpy as np


class Ellipse:
    def __init__(self, theta=None):
        self.theta = theta

    def evaluate(self, parameters):
        assert self.theta is not None, "theta is not initialized"

        x = self.theta[0]*tf.cos(parameters) \
            + self.theta[1]*tf.sin(parameters) + self.theta[2]
        y = self.theta[3]*tf.cos(parameters) \
            + self.theta[4]*tf.sin(parameters) + self.theta[5]

        return tf.stack([x, y], axis=-1)

    def set_theta(self, theta):
        self.theta = theta

    def plot(self, ax):
        points_on_model = self.evaluate(tf.linspace(0., 2*np.pi, 100))
        ax.plot(points_on_model[:, 0], points_on_model[:, 1])
