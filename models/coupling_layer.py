#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu August 23 12:00:57 2018
@author: l.faury
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Reshape
from models.mask import EvenMask
from models.mlp import MultiLayerPerceptron


class CouplingLayer(object):
    ''' Affine coupling layer class '''

    def __init__(self, mode, input_tensor, mask, width_model, translation_model):
        """ Init
        Args:
            mode: string, 'feed-forward' (feed-forward mode) or 'feed-backward' (feed-backward mode)
            input_tensor: tensor like, input tensor of the coupling layer
            mask: binary mask object
            width_model: keras model for the width network
            translation_model: keras model for the translation network
        """
        self.mode = mode
        self.size = int(input_tensor.get_shape()[1])

        # mask
        # --------------
        self.mask = mask
        # --------------

        # width and translation
        # --------------
        w = width_model
        t = translation_model
        # --------------

        # mask the input
        # --------------
        input_masked = Lambda(lambda x: tf.multiply(x, self.mask))(input_tensor)
        # --------------

        # feed-forward mode
        # ------------------
        if self.mode == 'feed_forward':
            # sample
            # ------------------
            forward_layer = Lambda(lambda x: tf.multiply(x[0], tf.exp(w([x[1]]))) + t([x[1]]))
            width_translation = forward_layer((input_tensor, input_masked))
            output = Lambda(lambda x: x[0] + tf.multiply(x[1], 1 - self.mask))([input_masked, width_translation])
            # ------------------
            # Jacobian computation
            # ------------------
            # det_jac = Lambda(lambda x: tf.exp(tf.reduce_sum(tf.multiply(1 - self.mask, w([x])), axis=1)))([input_masked])
            # det_jac = Reshape((1,))(det_jac)
            # det_jac_test = tf.matrix_determinant(
            #     tf.stack([tf.gradients(output[:, idx], input_tensor)[0] for idx in range(self.size)], axis=1))
            # det_jac_test = tf.reshape(det_jac_test, (-1, 1))
        # --------------

        # feed-backward mode
        # --------------
        elif self.mode == 'feed_backward':
            # sample
            # ------------------
            backward_layer = Lambda(lambda x: tf.multiply(tf.exp(-w([x[1]])), x[0]-t([x[1]])))
            width_translation = backward_layer([input_tensor, input_masked])
            output = Lambda(lambda x: x[0] + tf.multiply(x[1], 1 - self.mask))([input_masked, width_translation])
            # ------------------
            # Jacobian computation
            # ------------------
            log_det_jac = Lambda(lambda x: -tf.reduce_sum(tf.multiply(1 - self.mask, w([x])), axis=1))([input_masked])
            log_det_jac = Reshape((1,))(log_det_jac)
            det_jac_test = tf.log(tf.abs(tf.matrix_determinant(
                tf.stack([tf.gradients(output[:, idx], input_tensor)[0] for idx in range(self.size)], axis=1))))
            det_jac_test = tf.reshape(det_jac_test, (-1, 1))
            self.log_det_jac = log_det_jac
            self.det_jac_test = det_jac_test
        # --------------
        else:
            raise ValueError('Unknown mode for coupling layer')

        self.output = output


################################################################################
#                             UNIT TESTING                                     #
################################################################################
if __name__ == '__main__':
    noise_size = 2
    tf.set_random_seed(0)
    np.random.seed(0)

    w_in = Input(shape=(noise_size,), name='w_in')
    w_out = MultiLayerPerceptron(input_tensor=w_in,
                                 output_size=noise_size,
                                 name_scope='test_scope',
                                 name='width_network').out
    w = Model([w_in], w_out)
    t_in = Input(shape=(noise_size,), name='t_in')
    t_out = MultiLayerPerceptron(input_tensor=t_in,
                                 output_size=noise_size,
                                 name_scope='test_scope',
                                 name='translation_network').out
    t = Model([t_in], t_out)

    feed_forward_input = Input(shape=(noise_size,),name='ff_in')
    feed_backward_input =  Input(shape=(noise_size,),name='fb_in')

    mask = EvenMask(noise_size)
    mask_variable = tf.Variable(mask(), dtype=tf.float32, trainable=False)
    forward_coupling_layer = CouplingLayer(mode='feed_forward',
                                           mask=mask_variable,
                                           input_tensor=feed_forward_input,
                                           width_model=w,
                                           translation_model=t)
    backward_coupling_layer = CouplingLayer(mode='feed_backward',
                                            mask=mask_variable,
                                            input_tensor=feed_backward_input,
                                            width_model=w,
                                            translation_model=t)

    # test feed forward
    z = np.random.normal(size=(10, noise_size))

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        # feed-forward
        feed_dict = {feed_forward_input: z}
        x = session.run(forward_coupling_layer.output, feed_dict=feed_dict)
        print(x)
        print('\n')
        # round trip, expect 0
        feed_dict = {feed_backward_input: x}
        zz = session.run(backward_coupling_layer.output, feed_dict=feed_dict)
        print(np.round(10**6*(z-zz)))
        print('\n')

        # jacobian test, expect 0
        forward_jac, forward_jac_test = session.run([forward_coupling_layer.det_jac, forward_coupling_layer.det_jac_test],
                                                    feed_dict={feed_forward_input: z})
        print(forward_jac-forward_jac_test)

        backward_jac, backward_jac_test = session.run([backward_coupling_layer.det_jac, backward_coupling_layer.det_jac_test],
                                                    feed_dict={feed_backward_input: z})
        print(backward_jac-backward_jac_test)
    # ============== #
