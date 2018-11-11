
import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda
import numpy as np
import logging
from utils.utils import get_or_default


class MultiLayerPerceptron(object):
    """ MLP class """
    def __init__(self,
                 input_tensor,
                 output_size,
                 name_scope=None,
                 name=None,
                 config={}):
        # get default params and warns user
        # -----------
        logger = logging.getLogger(self.__class__.__name__+":"+str(name_scope)+':'+str(name))
        hidden_sizes = get_or_default(config, 'hidden_sizes', [32, 32], logger)
        hidden_facq = get_or_default(config, 'hidden_facq', 'tanh', logger)
        output_facq = get_or_default(config, 'output_facq', 'linear', logger)
        scale_output = get_or_default(config, 'scale_output', True, logger)
        # -----------
        # builds model
        # -----------
        with tf.name_scope(name_scope):
            x = input_tensor
            for h in hidden_sizes:
                x = Dense(h, activation=hidden_facq)(x)
            out = Dense(output_size, activation=output_facq)(x)
            if scale_output:
                scale = tf.Variable(0.1*np.ones((output_size,)), dtype=tf.float32)
                out = Lambda(lambda x: scale*x)(out)
        # -----------
        self.out = out
