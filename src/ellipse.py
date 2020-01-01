import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def get_points_on_ellipse(params_ellipse, params_points):
    """Returns points on ellipse from given parameters.

    Arguments
    ---------
    params_ellipse: tf.Tensor; (6,); tf.float32
        ellipse equation parameters.
    params_points: tf.Tensor; (n_points,); tf.float32
        curve parameter which lies in [0, 2*np.pi].

    Returns
    -------
    points_on_ellipse: tf.Tensor; (n_points, 2); tf.float32
        2D points on ellipse based on given parameters.
    """
    x = params_ellipse[0] * tf.cos(params_points) \
        + params_ellipse[1] * tf.sin(params_points) \
        + params_ellipse[2]
    y = params_ellipse[3] * tf.cos(params_points) \
        + params_ellipse[4] * tf.sin(params_points) \
        + params_ellipse[5]
    points_on_ellipse = tf.stack([x, y], axis=-1)

    return points_on_ellipse


if __name__ == "__main__":
    # initialize parameters
    params_ellipse = tf.random.uniform((6,))
    params_points = tf.linspace(0., 2*np.pi, 100)  # 100 points

    points_on_ellipse = get_points_on_ellipse(params_ellipse, params_points)

    plt.plot(points_on_ellipse[:, 0], points_on_ellipse[:, 1])
    plt.show()
