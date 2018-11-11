
import numpy as np
import tensorflow as tf
from models.mask import RandomMask, EvenMask, OddMask
from models.mlp import MultiLayerPerceptron
from models.additive_coupling_layer import AdditiveCouplingLayer
from models.lower_triangular import LowerTriangularLayer, LowerTriangularParameters
from models.invertible_activation import InvertibleIdentity
from tensorflow.keras.layers import Lambda, Concatenate
import unittest as ut
import matplotlib.pyplot as plt
import logging
from utils.utils import get_or_default
from utils.tensorflow_utils import create_model


class NICENetwork(object):
    """ Invertible Neural Network """
    def __init__(self,
                 config,
                 output_size,
                 latent_vector_input,
                 data_vector_input,
                 name_scope):
        """
        :param config: dictionary
        :param output_size: int
        :param latent_vector_input: generalized tensorflow input
        :param name_scope: string
        """
        logger = logging.getLogger(self.__class__.__name__+":"+str(name_scope))
        # get config
        # ---------
        depth = get_or_default(config, 'depth', 3, logger)
        translation_network_config = get_or_default(config, 'translation_network', {}, logger)
        translation_networks_args = {'output_size': output_size,
                                     'name_scope': name_scope,
                                     'name': 'translation_network',
                                     'config': translation_network_config}
        # ---------

        # affine transformation of the input
        triangular_params = LowerTriangularParameters(output_size, name_scope, trainable_bias=False)
        # generate masks, width and translation networks
        # ---------
        masks = []
        t_model = []
        for d in range(depth):
            # alternating random masks for the coupling layers
            if d % 2 == 0:
                mask = EvenMask(output_size)
            else:
                mask = OddMask(output_size)
            with tf.name_scope(name_scope):
                m = tf.Variable(mask(), dtype=tf.float32, trainable=False, name='mask')
            masks.append(m)
            # create t
            t_model.append(create_model(MultiLayerPerceptron,
                                        input_size=output_size,
                                        model_args=translation_networks_args))
        # ---------

        # build the full sample model by stacking coupling layers
        # ---------
        # feed-forward
        mode = 'feed_forward'
        forward_layers = list()
        forward_layers.append(LowerTriangularLayer(input_tensor=latent_vector_input,
                                                   lower_triangular_matrix=triangular_params.lower_triangular_matrix,
                                                   bias=triangular_params.bias,
                                                   mode=mode,
                                                   invertible_facq=InvertibleIdentity()))
        for d in range(0, depth):
            forward_layers.append(AdditiveCouplingLayer(mode=mode,
                                                        input_tensor=forward_layers[d].output,
                                                        mask=masks[d],
                                                        translation_model=t_model[d]))
        self._data_vector = forward_layers[depth].output

        # feed backward
        mode = 'feed_backward'
        backward_layers = list()
        backward_layers.append(AdditiveCouplingLayer(mode=mode,
                                                     input_tensor=data_vector_input,
                                                     mask=masks[depth-1],
                                                     translation_model=t_model[depth-1]))
        for d in range(1, depth):
            backward_layers.append(AdditiveCouplingLayer(mode=mode,
                                                         input_tensor=backward_layers[d-1].output,
                                                         mask=masks[depth-1-d],
                                                         translation_model=t_model[depth-1-d]))
        backward_layers.append(LowerTriangularLayer(input_tensor=backward_layers[depth-1].output,
                                                    lower_triangular_matrix=triangular_params.lower_triangular_matrix,
                                                    bias=triangular_params.bias,
                                                    mode=mode,
                                                    invertible_facq=InvertibleIdentity()))
        self._latent_vector = backward_layers[depth].output

        # ---------

        # build the likelihood model
        # ---------
        concat_log_det_jac = Concatenate(axis=1)([backward_layers[c].log_det_jacobian for c in range(depth+1)])

        self._log_det_jac = Lambda(lambda x: tf.reduce_sum(x, axis=1))(concat_log_det_jac)
        self._log_det_jac_test = tf.log(tf.matrix_determinant(tf.stack(
            [tf.gradients(self._latent_vector[:, idx], data_vector_input)[0] for idx in
             range(output_size)], axis=1)))
        self._llh = Lambda(
            lambda x: -0.5 * output_size * np.log(2 * np.pi) - 0.5 * tf.reduce_sum(tf.square(x[0]), axis=1) + \
                      x[1])([self._latent_vector, self._log_det_jac])
        # ---------

    @property
    def data_vector(self):
        return self._data_vector

    @property
    def latent_vector(self):
        return self._latent_vector

    @property
    def log_det_jac(self):
        return self._log_det_jac

    @property
    def log_det_jac_test(self):
        return self._log_det_jac_test

    @property
    def llh(self):
        return self._llh


# ################################################################################
# #                             UNIT TESTING                                     #
# ################################################################################

# class TestConditionalInvNet(tf.test.TestCase):
#
#     def setUp(self):
#         tf.reset_default_graph()
#         tf.set_random_seed(0)
#         np.random.seed(0)
#         self.output_size = 2
#         self.batch_size = 10000
#         name_scope = 'test_conditional_invnet'
#
#         self.latent_input = tf.placeholder(dtype=tf.float32, shape=(None, self.output_size), name='latent')
#         self.data_input = tf.placeholder(dtype=tf.float32, shape=(None, self.output_size), name='data')
#
#         self.invertible_network = NICENetwork(config={},
#                                               output_size= self.output_size,
#                                               name_scope=name_scope,
#                                               data_vector_input= self.data_input,
#                                               latent_vector_input= self.latent_input)
#         self.latent_vector_val = np.random.normal(0, 1, (self.batch_size, self.output_size))
#
#
#     def test_conditional_invnet_sampling(self):
#         print('Testing sampling \n --------')
#         with tf.Session() as session:
#             session.run(tf.global_variables_initializer())
#             sample = session.run(self.invertible_network.data_vector,
#                                  feed_dict={self.latent_input: self.latent_vector_val})
#         plt.scatter(self.latent_vector_val[:, 0], self.latent_vector_val[:, 1], c='blue')
#         plt.scatter(sample[:, 0], sample[:, 1], c='red')
#         plt.show()
#
#     def test_conditional_invnet_invertibility(self):
#         print('\n Testing inversion, expects 0 \n --------')
#         with tf.Session() as session:
#             session.run(tf.global_variables_initializer())
#             sample = session.run(self.invertible_network.data_vector,
#                                  feed_dict={self.latent_input: self.latent_vector_val})
#             recovered_latent_vector = session.run(self.invertible_network.latent_vector,
#                                                   feed_dict={self.data_input: sample})
#
#         max_element = np.max([np.abs(recovered_latent_vector-self.latent_vector_val)])
#         precision = 5
#         print('recovered_latent_vector and initial latent_vector should be equal')
#         print('precision used 10-' + str(precision))
#         self.assertAlmostEqual(max_element, 0, precision)
#
#     def test_conditional_invnet_detjac(self):
#         # testing detjac
#         print('\n Testing jacobian, expects 0s \n --------')
#         data = np.random.normal(0, 1, (self.batch_size, self.output_size))
#         with tf.Session() as session:
#             session.run(tf.global_variables_initializer())
#             sample = session.run(self.invertible_network.data_vector,
#                                  feed_dict={self.latent_vector: self.latent_vector_val})
#
#             djac = session.run(self.invertible_network.log_det_jac,
#                                feed_dict={self.data_vector: sample})
#             djactest = session.run(self.invertible_network.log_det_jac_test,
#                                     feed_dict={self.data_vector: sample})
#         print('djac - djactest should be equal')
#         precision = 5
#         print('precision used 10-' + str(precision))
#         self.assertAlmostEqual(np.max(np.abs(djac - djactest)), 0, precision)
#
#     def test_conditional_invnet_log_likelihood(self):
#         print('\n Testing log-likelihood \n --------')
#         size = int(np.sqrt(self.batch_size))
#         l = np.linspace(-4, 4, size)
#         X, Y = np.meshgrid(l, l)
#         data = np.transpose(np.array([X.flatten(), Y.flatten()]))
#         with tf.Session() as session:
#             session.run(tf.global_variables_initializer())
#
#             llkdx = session.run(self.invertible_network.llh,
#                                 feed_dict={self.data_vector: data})
#
#             sample = session.run(self.invertible_network.data_vector,
#                                  feed_dict={self.latent_vector: self.latent_vector_val})
#
#             lkdx = np.reshape(np.exp(llkdx), (size, size))
#         plt.figure()
#         CS = plt.contour(X, Y, lkdx)
#         plt.colorbar(CS, shrink=0.8, extend='both')
#         plt.scatter(sample[:, 0], sample[:, 1], c='red', marker='+')
#         plt.show()
#
# if __name__ == "__main__":
#     ut.main()
