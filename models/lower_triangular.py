

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda, Add


class LowerTriangularParameters(object):
    def __init__(self,
                 dim,
                 name_scope,
                 tol=1e-12,
                 trainable_bias=True):
        with tf.name_scope(name_scope):
            log_eigen_values = tf.Variable(np.zeros(dim), dtype=tf.float32)
            safe_eigen_values = tf.maximum(tol, tf.exp(log_eigen_values))
            extra_diagonal_values = tf.Variable(np.zeros((dim, dim)), dtype=tf.float32)
            lower_triangular_without_diag = tf.linalg.band_part(extra_diagonal_values, -1, 0)
            self.lower_triangular_matrix = tf.linalg.set_diag(lower_triangular_without_diag, safe_eigen_values)
            self.bias = tf.Variable(np.zeros((dim,)), dtype=tf.float32, trainable=trainable_bias)


class LowerTriangularLayer(object):
    def __init__(self,
                 input_tensor,
                 lower_triangular_matrix,
                 bias,
                 mode,
                 invertible_facq):
        if mode=='feed_forward':
            hidden = Lambda(lambda x: tf.transpose(tf.matmul(lower_triangular_matrix, tf.transpose(x)))+bias)(input_tensor)
            self.output = invertible_facq.forward(hidden)
        elif mode=='feed_backward':
            hidden = invertible_facq.backward(input_tensor)
            self.output = Lambda(lambda x: tf.transpose(tf.linalg.triangular_solve(lower_triangular_matrix, tf.transpose(x-bias))))(hidden)
            self.support = Lambda(lambda x: tf.ones((tf.shape(x)[0],1)))(input_tensor)
            self.latent_prob =  Lambda(lambda x: -tf.reduce_sum(tf.log(tf.diag_part(lower_triangular_matrix)))*x)(self.support)
            self.normalizing_factor = invertible_facq.log_det_jacobian(input_tensor)
            self.log_det_jacobian = Add()([self.normalizing_factor, self.latent_prob])

if __name__=='__main__':

    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    from models.invertible_activation import create_invertible_activation_function
    inputs = Input(shape=(10,))
    params = LowerTriangularParameters(10, 'test')
    layer = LowerTriangularLayer(inputs, params.lower_triangular_matrix, params.bias, 'feed_backward',
                                 create_invertible_activation_function('sigmoid'))
    model = Model(inputs=inputs, outputs=layer.log_det_jacobian)