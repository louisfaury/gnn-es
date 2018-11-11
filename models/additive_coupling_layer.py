
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Reshape
from models.mask import EvenMask, OddMask
from models.mlp import MultiLayerPerceptron


class AdditiveCouplingLayer(object):
    ''' Affine coupling layer class '''

    def __init__(self, mode, input_tensor, mask, translation_model):
        """ Init
        Args:
            mode: string, 'feed-forward' (feed-forward mode) or 'feed-backward' (feed-backward mode)
            input_tensor: tensor like, input tensor of the coupling layer
            mask: binary mask object
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
            forward_layer = Lambda(lambda x: x[0] + t([x[1]]))((input_tensor, input_masked))
            output = Lambda(lambda x: x[0] + tf.multiply(x[1], 1 - self.mask))([input_masked, forward_layer])
        # --------------

        # feed-backward mode
        # --------------
        elif self.mode == 'feed_backward':
            # sample
            # ------------------
            backward_layer = Lambda(lambda x: x[0] - t([x[1]]))([input_tensor, input_masked])
            output = Lambda(lambda x: x[0] + tf.multiply(x[1], 1 - self.mask))([input_masked, backward_layer])
            # ------------------
            # Jacobian computation
            # ------------------
            log_det_jacobian = Lambda(lambda x: tf.zeros(shape=(tf.shape(x)[0],)))(input_tensor)
            log_det_jacobian = Reshape((1,))(log_det_jacobian)
            log_det_jac_test = tf.log(tf.matrix_determinant(
                tf.stack([tf.gradients(output[:, idx], input_tensor)[0] for idx in range(self.size)], axis=1)))
            log_det_jac_test = tf.reshape(log_det_jac_test, (-1, 1))
        # --------------
            self.log_det_jacobian = log_det_jacobian
            self.log_det_jac_test = log_det_jac_test
        else:
            raise ValueError('Unknown mode for coupling layer')

        self.output = output



################################################################################
#                             UNIT TESTING                                     #
################################################################################
if __name__ == '__main__':
    noise_size = 2
    tf.set_random_seed(1)
    np.random.seed(1)

    t_in = Input(shape=(noise_size,), name='t_in')
    t_out = MultiLayerPerceptron(input_tensor=t_in,
                                 output_size=noise_size,
                                 name_scope='test_scope',
                                 name='translation_network').out
    t = Model([t_in], t_out)

    feed_forward_input = Input(shape=(noise_size,), name='ff_in')
    feed_backward_input = Input(shape=(noise_size,), name='fb_in')

    mask = EvenMask(noise_size)
    mask_variable = tf.Variable(mask(), dtype=tf.float32, trainable=False)
    forward_coupling_layer = AdditiveCouplingLayer(mode='feed_forward',
                                           mask=mask_variable,
                                           input_tensor=feed_forward_input,
                                           translation_model=t)
    backward_coupling_layer = AdditiveCouplingLayer(mode='feed_backward',
                                            mask=mask_variable,
                                            input_tensor=feed_backward_input,
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
        print(np.round(10 ** 6 * (z - zz)))
        print('\n')

        # jacobian test, expect 0
        backward_jac, backward_jac_test = session.run(
            [backward_coupling_layer.log_det_jacobian, backward_coupling_layer.log_det_jac_test],
            feed_dict={feed_backward_input: z})
        print(backward_jac - backward_jac_test)
    # ============== #