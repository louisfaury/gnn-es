
from abc import ABC, abstractmethod


class BaseSampler(ABC):
    def __init__(self, distribution, sample_size):
        self.distribution = distribution
        self.sample_size = sample_size

    @abstractmethod
    def sample(self):
        pass