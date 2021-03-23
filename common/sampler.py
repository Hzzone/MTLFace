import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


class RandomSampler(Sampler):

    def __init__(self, dataset, batch_size=0, num_iter=None, restore_iter=0,
                 weights=None, replacement=True, seed=0):
        super(RandomSampler, self).__init__(dataset)
        self.dist = dist.is_initialized()
        if self.dist:
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0
        self.dataset = dataset
        self.batch_size = batch_size * self.num_replicas
        self.num_samples = num_iter * self.batch_size
        self.restore = restore_iter * self.batch_size
        self.weights = weights
        self.replacement = replacement
        self.seed = seed

    def __iter__(self):
        # deterministically shuffle
        g = torch.Generator()
        g.manual_seed(self.seed)
        if self.weights is None:
            n = len(self.dataset)
            n = n - n % self.batch_size
            epochs = self.num_samples // n + 1
            indices = []
            for e in range(epochs):
                g = torch.Generator()
                g.manual_seed(self.seed + e)
                # drop last
                indices.extend(torch.randperm(n, generator=g).tolist()[:n])
            indices = indices[:self.num_samples]
            # indices = torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64, generator=g).tolist()
        else:
            indices = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=g).tolist()

        # subsample
        indices = indices[self.restore + self.rank:self.num_samples:self.num_replicas]

        return iter(indices)

    def __len__(self):
        return self.num_samples - self.restore
