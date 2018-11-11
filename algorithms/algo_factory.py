
from algorithms.vanilla_search_gradients import VanillaSearchGradients
from algorithms.adaptive_search_gradients import AdaptiveVanillaSearchGradients
from algorithms.trust_region_search_gradients import TrustRegionSearchGradients
from algorithms.natural_search_gradients import NaturalSearchGradients


def create_algorithm(config, distribution):
    name = config["name"]
    if name == 'vsg':
        algo = VanillaSearchGradients(config, distribution)
    elif name == 'a-vsg':
        algo = AdaptiveVanillaSearchGradients(config, distribution)
    elif name == 'trsg':
        algo = TrustRegionSearchGradients(config, distribution)
    elif name == 'nsg':
        algo = NaturalSearchGradients(config, distribution)
    else:
        raise NotImplementedError("Unknown algorithm with name :" + name)
    return algo