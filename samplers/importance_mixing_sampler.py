
from samplers.base_sampler import BaseSampler
import numpy as np


class ImportanceMixingSampler(BaseSampler):
    def __init__(self, config, distribution, sample_size):
        super().__init__(distribution, sample_size)
        self.alpha = config["refresh_rate"]
        self.buffer = list()
        self.distribution_copy = self.distribution.build_copy()

    def sample(self):
        if len(self.buffer)==0:
            sample = self.distribution.sample(self.sample_size)
            self.buffer += list(sample)
            self.distribution_copy.copy_from_distribution(self.distribution)
            sampled_from_past = 0

        else:
            reshaped_buffer = np.reshape(self.buffer, (-1, self.distribution.output_size))
            old_pdf = self.distribution_copy.pdf(reshaped_buffer)
            new_pdf = self.distribution.pdf(reshaped_buffer)
            ip_ratio = new_pdf / (old_pdf + 1e-6)
            acceptance_prob = np.minimum(1, (1-self.alpha)*ip_ratio)
            keep = np.random.binomial(1, acceptance_prob)
            idx = np.where(keep)
            sample = reshaped_buffer[idx]
            accepted_samples = np.size(idx)
            sampled_from_past = accepted_samples

            # accept for new history
            while accepted_samples < self.sample_size:
                x = self.distribution.sample(1)
                old_pdf = np.asscalar(self.distribution_copy.pdf(x))
                new_pdf = np.asscalar(self.distribution.pdf(x))
                ip_ratio = old_pdf/(new_pdf+1e-8)
                acceptance_prob = np.max([self.alpha, 1-ip_ratio])
                if np.random.uniform(0,1) < acceptance_prob:
                    sample = np.vstack((sample, x))
                    accepted_samples += 1

        return np.reshape(np.array(sample), (self.sample_size, self.distribution.output_size)), self.sample_size-sampled_from_past