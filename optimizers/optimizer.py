
import tensorflow as tf


def create_optimizer(name, lr=None):
    """
    :param name: string
    :param lr: float, learning rate
    :return: optimizer
    """
    if name == 'adam':
        optimizer = tf.train.AdamOptimizer(lr)
    elif name == 'gd':
        optimizer = tf.train.GradientDescentOptimizer(lr)
    elif name == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(lr)
    else:
        raise NotImplementedError('Unknown optimizer with name :'+name)
    return optimizer
