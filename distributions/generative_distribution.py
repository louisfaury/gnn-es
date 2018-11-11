
import tensorflow as tf
from abc import abstractmethod
from distributions.base_distribution import Distribution
from utils.utils import get_or_default
from tensorflow.keras.models import Model


class GenerativeDistribution(Distribution):
    def __init__(self, output_size, name, name_scope, config):
        super().__init__(output_size, name, name_scope)
        self.config = config
        sample_model = Model(inputs=[self.latent_input_layer], outputs=self.invertible_network.data_vector)
        llh_model = Model(inputs=[self.data_input_layer], outputs=self.invertible_network.llh)

        # sampling operation
        self.data_vector_out = sample_model(self.latent_vector)

        # likelihood operations
        self._log_pdf = llh_model([self.data_vector_in])
        self._pdf = tf.exp(self._log_pdf)

        # re-sampling operations (for std, entropy, kl, ..)
        self.monte_carlo_batch_size = get_or_default(config, "monte_carlo_batch_size", 200, self.log)
        latent_vector_resampled = tf.random_normal(shape=(self.monte_carlo_batch_size, self.output_size))
        data_vector_resampled = sample_model([latent_vector_resampled])
        llh_resampled = llh_model(data_vector_resampled)

        # std op
        var = tf.nn.moments(data_vector_resampled, axes=[0])[1]
        self._std = tf.reduce_prod(tf.sqrt(var))

        # entropy op
        self._entropy = -tf.reduce_mean(llh_resampled)

        # for kl op
        self.llh_model = llh_model
        self.llh_resampled = llh_resampled
        self.data_vector_resampled = data_vector_resampled

        # copy op
        self.build_copy_op()

    def build_kl(self, old_distribution):
        old_resampled_actions = tf.stop_gradient(old_distribution.data_vector_resampled)
        old_log_pdf_on_old_actions = tf.stop_gradient(old_distribution.llh_resampled)
        new_log_pdf_on_old_actions = self.llh_model(old_resampled_actions)
        kl = tf.reduce_mean(old_log_pdf_on_old_actions - new_log_pdf_on_old_actions)
        self._kl = kl
        return kl

    @abstractmethod
    def build_copy(self, name_scope):
        pass