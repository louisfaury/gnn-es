
from distributions.base_distribution import Distribution
import tensorflow as tf
import numpy as np


class Gaussian(Distribution):
    """ Full Gaussian distribution class """
    def __init__(self, output_size, name, name_scope):
        super().__init__(output_size, name, name_scope)

        # creating variables: x = Az+b
        # ----------------
        with tf.name_scope(name_scope):
            log_std_eigen_values = tf.Variable(np.zeros(output_size), name='sigma_ev', dtype=tf.float32)
            std_extra_diagonal_values = tf.Variable(np.zeros((output_size, output_size)), dtype=tf.float32)
            std_lower_triangular_without_diag = tf.linalg.band_part(std_extra_diagonal_values, -1, 0)
            std_safe_eigen_values = tf.maximum(1e-8, tf.exp(log_std_eigen_values))
            std_lower_triangular = tf.linalg.set_diag(std_lower_triangular_without_diag, std_safe_eigen_values)
            b = tf.Variable(np.zeros((output_size,)), name='bias', dtype=tf.float32)
        # ----------------
        std = std_lower_triangular
        self.tol = tf.constant(1e-9*np.eye(output_size), dtype=tf.float32)

        # sampling operation
        self.data_vector_out = tf.transpose(tf.matmul(std, self.latent_vector, transpose_b=True)) + b

        # likelihood operation
        x = self.data_vector_in
        std_log_det = tf.reduce_sum(tf.log(std_safe_eigen_values))
        log_normalizing_factor = -0.5*(tf.constant(output_size*np.log(2*np.pi), dtype=tf.float32)) - std_log_det
        z = tf.transpose(tf.linalg.triangular_solve(std, tf.transpose(x-b), lower=True))
        log_exp = -0.5*tf.reduce_sum(z*z, axis=1)
        self._log_pdf = log_exp + log_normalizing_factor
        self._pdf = tf.exp(self._log_pdf, name='pdf')

        # statistics
        self.std = std
        self.std_log_det = std_log_det
        self.mean = b

        # entropy op
        self._entropy = tf.constant(0.5*output_size*np.log(2*np.pi*np.e), dtype=tf.float32) + std_log_det

        # copy op
        self.build_copy_op()

    def build_copy(self, name_scope='distribution_copy'):
        return Gaussian(output_size=self.output_size,
                        name='gaussian',
                        name_scope=name_scope)

    def build_kl(self, old_distribution):
        old_std = tf.stop_gradient(old_distribution.std)
        old_mean = tf.stop_gradient(old_distribution.mean)
        old_std_log_det = tf.stop_gradient(old_distribution.std_log_det)
        old_sigma = tf.matmul(old_std, old_std, transpose_b=True)
        old_sigma = 0.5*(tf.transpose(old_sigma) + old_sigma) + self.tol

        sigma = tf.matmul(self.std, self.std, transpose_b=True)
        sigma = 0.5*(tf.transpose(sigma)+sigma) + self.tol

        inv_mat = tf.linalg.solve(sigma, old_sigma)
        trace = tf.linalg.trace(inv_mat)
        mean_diff = tf.expand_dims(self.mean-old_mean, 1)
        z = tf.linalg.triangular_solve(self.std, mean_diff, lower=True)
        dot_prod = tf.reduce_sum(z*z)
        log_det_ratio = 2*self.std_log_det - 2*old_std_log_det
        kl = 0.5*(trace + dot_prod + log_det_ratio - self.output_size)
        self._kl = kl
        return kl
