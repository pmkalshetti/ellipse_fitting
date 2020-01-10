import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
from ellipse import get_points_on_ellipse


def generate_data_from_ellipse(
        params_ellipse=[3, 2, 5, 4, 5, 7.],
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
    params_ellipse = tf.convert_to_tensor(params_ellipse)
    params_points = tf.linspace(0., 2*np.pi*fraction, n_data)
    data = get_points_on_ellipse(params_ellipse, params_points) \
        + tf.random.normal((n_data, 2), stddev=stddev_noise)

    return data


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
    data = generate_data_from_ellipse()

    # plot
    plt.scatter(
        data[:, 0], data[:, 1],
        facecolors="none", edgecolors="black", s=20
    )
    plt.title("Data sampled from ellipse")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(0, 15)
    plt.ylim(0, 15)
    plt.show()
