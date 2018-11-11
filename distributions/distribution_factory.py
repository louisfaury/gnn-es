
from distributions.ddm import DeepDensityModel
from distributions.gaussian import Gaussian
from distributions.nice_distribution import NICE
from distributions.nvp import RealNVP


def create_distribution(config, output_size):
    distribution_name = config["name"].lower()
    name_scope = 'search_distribution'
    if distribution_name == 'gaussian':
        distribution = Gaussian(output_size=output_size,
                                name=distribution_name,
                                name_scope=name_scope)
    elif distribution_name == 'nvp':
        distribution = RealNVP(config=config,
                               output_size=output_size,
                               name=distribution_name,
                               name_scope=name_scope)
    elif distribution_name == 'ddm':
        distribution = DeepDensityModel(config=config,
                                        output_size=output_size,
                                        name=distribution_name,
                                        name_scope=name_scope)
    elif distribution_name == 'nice':
        distribution = NICE(config=config,
                            output_size=output_size,
                            name=distribution_name,
                            name_scope=name_scope)
    else:
        raise ValueError('Unknown distribution name :', distribution_name)
    return distribution
