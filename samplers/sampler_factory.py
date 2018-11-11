
from samplers.plain_sampler import PlainSampler
from samplers.importance_mixing_sampler import ImportanceMixingSampler

def create_sampler(config, distribution):
    sampler_name = config["name"]
    sample_size = config["pop_size"]

    if sampler_name == 'plain':
        sampler = PlainSampler(distribution, sample_size)
    elif sampler_name == 'importance_mixing':
        sampler = ImportanceMixingSampler(config, distribution, sample_size)
    else:
        raise NotImplementedError('Unknown sampler name')

    return sampler