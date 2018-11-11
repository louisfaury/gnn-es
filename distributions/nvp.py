
from models.nvp_network import NonVolumePreservingNetwork
from distributions.generative_distribution import GenerativeDistribution
from tensorflow.keras.layers import Input


class RealNVP(GenerativeDistribution):
    def __init__(self,
                 output_size,
                 config,
                 name_scope,
                 name):
        # build the invertible network model
        # ------------
        invertible_network_config = config["invertible_network"]
        self.latent_input_layer = Input(shape=(output_size,))
        self.data_input_layer = Input(shape=(output_size,))
        # ------------
        self.invertible_network = NonVolumePreservingNetwork(config=invertible_network_config,
                                                             output_size=output_size,
                                                             latent_vector_input=self.latent_input_layer,
                                                             data_vector_input=self.data_input_layer,
                                                             name_scope=name_scope)
        # ------------

        # Generative Distribution
        super().__init__(output_size, name, name_scope, config)

    def build_copy(self, name_scope='distribution_copy'):
        return RealNVP(output_size=self.output_size,
                       config=self.config,
                       name_scope=name_scope,
                       name='nvp')