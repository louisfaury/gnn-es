
import tensorflow as tf
from distributions.generative_distribution import GenerativeDistribution
from models.invertible_mlp import InvertibleMultiLayerPerceptron
from tensorflow.keras.layers import Input


class DeepDensityModel(GenerativeDistribution):
    def __init__(self, output_size, config, name_scope, name):
        # build the invertible mlp
        # ------------
        self.latent_input_layer = Input(shape=(output_size,))
        self.data_input_layer = Input(shape=(output_size,))
        # ------------
        self.invertible_network = InvertibleMultiLayerPerceptron(config=config["invertible_network"],
                                                                 output_size=output_size,
                                                                 latent_input=self.latent_input_layer,
                                                                 data_input=self.data_input_layer,
                                                                 name_scope=name_scope)
        # ------------

        # Generative Distribution
        super().__init__(output_size, name, name_scope, config)

    def build_copy(self, name_scope='distribution_copy'):
        return DeepDensityModel(output_size=self.output_size,
                                config=self.config,
                                name_scope=name_scope,
                                name='ddm')

    def build_kl(self, old_distribution):
        old_resampled_actions = tf.stop_gradient(old_distribution.data_vector_resampled)
        old_log_pdf_on_old_actions = tf.stop_gradient(old_distribution.llh_resampled)
        new_log_pdf_on_old_actions = self.llh_model(old_resampled_actions)
        kl = tf.reduce_mean(old_log_pdf_on_old_actions-new_log_pdf_on_old_actions)
        self._kl = kl
        return kl