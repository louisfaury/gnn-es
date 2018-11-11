
import numpy as np
import logging
import tensorflow as tf
from models.lower_triangular import LowerTriangularLayer, LowerTriangularParameters
from models.invertible_activation import create_invertible_activation_function
from utils.utils import get_or_default
from tensorflow.keras.layers import Lambda, Concatenate


class InvertibleMultiLayerPerceptron(object):

    def __init__(self,
                 config,
                 output_size,
                 latent_input,
                 data_input,
                 name_scope):
        logger = logging.getLogger(self.__class__.__name__ + ":" + str(name_scope))
        depth = get_or_default(config, 'depth', 2, logger)
        activation_name = get_or_default(config, 'activation', 'tanh', logger)
        name_scope = name_scope

        # params
        # -------
        layers_params = []
        activations = []
        for d in range(depth):
            if d == depth-1:
                activations.append(create_invertible_activation_function('identity'))
            else:
                activations.append(create_invertible_activation_function(activation_name))
            layers_params.append(LowerTriangularParameters(output_size, name_scope))
        # -------

        # forward
        # -------
        forward_layers = []
        mode = 'feed_forward'
        forward_layers.append(
            LowerTriangularLayer(latent_input, layers_params[0].lower_triangular_matrix, layers_params[0].bias, mode, activations[0]))
        for d in range(1, depth):
            forward_layers.append(
                    LowerTriangularLayer(forward_layers[d-1].output, layers_params[d].lower_triangular_matrix,
                                         layers_params[d].bias, mode, activations[d]))
        self._data_vector = forward_layers[depth-1].output
        # -------

        # backward
        # -------
        backward_layers = []
        mode = 'feed_backward'
        backward_layers.append(
            LowerTriangularLayer(data_input, layers_params[depth - 1].lower_triangular_matrix,
                                 layers_params[depth-1].bias, mode, activations[depth-1]))
        for d in range(1, depth):
            backward_layers.append(
                LowerTriangularLayer(backward_layers[d-1].output, layers_params[depth-d-1].lower_triangular_matrix,
                                     layers_params[depth-d-1].bias, mode, activations[depth-d-1]))
        self._latent_vector = backward_layers[depth-1].output
        # -------

        # likelihood
        # -------
        concat_log_det_jacobian = Concatenate(axis=1)([backward_layers[c].log_det_jacobian for c in range(depth)])

        self._log_det_jac = Lambda(lambda x: tf.reduce_sum(x, axis=1))(concat_log_det_jacobian)
        self._log_det_jac_test = tf.log(tf.abs(tf.linalg.det((tf.stack(
            [tf.gradients(self._latent_vector[:, idx], data_input)[0] for idx in
             range(output_size)], axis=1)))))
        self._llh = Lambda(
            lambda x: -0.5 * output_size * np.log(2 * np.pi) - 0.5 * tf.reduce_sum(tf.square(x[0]), axis=1) + \
                      x[1])([self._latent_vector, self._log_det_jac])
        # -------

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


class TestConditionalInvNet(tf.test.TestCase):

    def setUp(self):
        tf.reset_default_graph()
        tf.set_random_seed(0)
        np.random.seed(0)
        self.output_size = 2
        self.batch_size = 100
        name_scope = 'test_ddm'

        self.latent_vector = tf.placeholder(dtype=tf.float32, shape=(None, self.output_size), name='latent')
        self.data_vector = tf.placeholder(dtype=tf.float32, shape=(None, self.output_size), name='data')

        self.invertible_network = InvertibleMultiLayerPerceptron(config={},
                                                                 output_size= self.output_size,
                                                                 name_scope=name_scope,
                                                                 data_input= self.data_vector,
                                                                 latent_input= self.latent_vector)
        self.latent_vector_val = np.random.normal(0, 1, (self.batch_size, self.output_size))

    def test_ddm_sampling(self):
        import matplotlib.pyplot as plt
        print('Testing sampling \n --------')
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            sample = session.run(self.invertible_network.data_vector,
                                 feed_dict={self.latent_vector: self.latent_vector_val})
            print(sample)
        plt.scatter(self.latent_vector_val[:, 0], self.latent_vector_val[:, 1], c='blue')
        plt.scatter(sample[:, 0], sample[:, 1], c='red')
        plt.show()

    def test_conditional_invnet_invertibility(self):
        print('\n Testing inversion, expects 0 \n --------')
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            sample = session.run(self.invertible_network.data_vector,
                                 feed_dict={self.latent_vector: self.latent_vector_val})

            recovered_latent_vector = session.run(self.invertible_network.latent_vector,
                                                  feed_dict={self.data_vector: sample})


        max_element = np.max([np.abs(recovered_latent_vector[i][j] - self.latent_vector_val[i][j]) for i in range(len(self.latent_vector_val)) for j in range(len(recovered_latent_vector[i]))])
        precision = 5
        print('recovered_latent_vector and initial latent_vector should be equal')
        print('precision used 10-' + str(precision))
        self.assertAlmostEqual(max_element, 0, precision)

    def test_conditional_invnet_detjac(self):
        # testing detjac
        print('\n Testing jacobian, expects 0s \n --------')
        data = np.random.normal(0, 1, (self.batch_size, self.output_size))
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            sample = session.run(self.invertible_network.data_vector,
                                 feed_dict={self.latent_vector: self.latent_vector_val})

            djac = session.run(self.invertible_network.log_det_jac,
                               feed_dict={self.data_vector: sample})
            djactest = session.run(self.invertible_network.log_det_jac_test,
                                    feed_dict={self.data_vector: sample})
        print('djac - djactest should be equal')
        precision = 5
        print('precision used 10-' + str(precision))
        self.assertAlmostEqual(np.max(np.abs(djac - djactest)), 0, precision)

    def test_conditional_invnet_log_likelihood(self):
        import matplotlib.pyplot as plt
        print('\n Testing log-likelihood \n --------')
        size = int(np.sqrt(self.batch_size))
        l = np.linspace(-0.98,0.98, size)
        X, Y = np.meshgrid(l, l)
        data = np.transpose(np.array([X.flatten(), Y.flatten()]))
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            llkdx = session.run(self.invertible_network.llh,
                                feed_dict={self.data_vector: data})

            sample = session.run(self.invertible_network.data_vector,
                                 feed_dict={self.latent_vector: self.latent_vector_val})

            lkdx = np.reshape(np.exp(llkdx), (size, size))
        plt.figure()
        print(lkdx)
        CS = plt.contour(X, Y, lkdx)
        # plt.colorbar(CS, shrink=0.8, extend='both')
        plt.scatter(sample[:, 0], sample[:, 1], c='red', marker='+')
        plt.show()

if __name__ == "__main__":
    ut.main()
