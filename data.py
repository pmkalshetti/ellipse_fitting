import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import Ellipse


class Data:
    def __init__(
            self,
            theta=tf.constant([3, 2, 5, 4, 5, 7.]),
            fraction=0.3,
            n_data=30,
            stddev=0.08
    ):
        self.theta = theta
        self.n_data = n_data

        self.model = Ellipse(self.theta)
        self.parameters = tf.linspace(0., 2*np.pi*fraction, self.n_data)
        self.samples = self._generate_samples(stddev=stddev)

    def _generate_samples(self, stddev):
        samples = self.model.evaluate(self.parameters)
        samples += tf.random.normal(samples.shape, stddev=stddev)

        return samples

    def plot(self, ax, xlim=(0, 15), ylim=(0, 15)):
        ax.scatter(self.samples[:, 0], self.samples[:, 1], s=10, label="data")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)


if __name__ == "__main__":
    tf.random.set_seed(1)

    data = Data()

    fig, ax = plt.subplots()
    data.plot(ax)
    plt.show()
