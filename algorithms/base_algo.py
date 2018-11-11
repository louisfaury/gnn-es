
from abc import ABC, abstractmethod


class BaseAlgo(ABC):
    def __init__(self, config, distribution):
        self.config = config
        self.distribution = distribution
        self.output_size = distribution.output_size

    @abstractmethod
    def fit(self, samples):
        pass