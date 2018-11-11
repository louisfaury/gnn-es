

from samplers.base_sampler import BaseSampler


class PlainSampler(BaseSampler):
    def __init__(self, distribution, sample_size):
        super().__init__(distribution, sample_size)

    def sample(self):
        return self.distribution.sample(self.sample_size), self.sample_size