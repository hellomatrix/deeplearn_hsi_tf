import numpy as np
import tensorflow as tf
import pdb


def makeGaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    filter = np.exp(-4 * np.log(2) * ((x - x0)**2 + (y - y0)**2) / fwhm**2)

    return filter / filter.sum()


def conv2d(x, W):
    W = tf.reshape(W, W.shape.as_list() + [1, 1])
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def tf_var2dic(var_list):
    var_dic = {}
    for var_list_i in var_list:
        key_ = var_list_i.name.split(':')[0]
        var_dic['/'.join(key_.split('/')[1:])] = var_list_i
    return var_dic


def xavier_init(fan_in, fan_out, function):
    if function is tf.nn.sigmoid:
        low = -4.0 * np.sqrt(6.0 / (fan_in + fan_out))
        high = 4.0 * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random_uniform(
            (fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)
    elif function is tf.nn.tanh:
        low = -1 * np.sqrt(6.0 / (fan_in + fan_out))
        high = 1 * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random_uniform(
            (fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    filter = makeGaussian(7)
    fig, ax = plt.subplots()
    ax.imshow(filter)
    plt.show()
