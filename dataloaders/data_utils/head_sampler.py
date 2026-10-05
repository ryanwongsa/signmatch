import numpy as np
import torch
from torch.utils.data import Sampler


class HeadSubsamplingSampler(Sampler):
    def __init__(self, labels, alpha=0.5):
        self.labels = np.asarray(labels)
        classes, counts = np.unique(self.labels, return_counts=True)
        median = np.median(counts)
        self.keep_prob = {
            c: min(1.0, (median / count) ** alpha) for c, count in zip(classes, counts)
        }
        self.size = int(sum(count * self.keep_prob[c] for c, count in zip(classes, counts)))

    def set_epoch(self, epoch):
        np.random.seed(epoch)
        torch.manual_seed(epoch)

    def __iter__(self):
        keep = np.vectorize(self.keep_prob.get)(self.labels)
        kept = np.nonzero(np.random.rand(len(self.labels)) < keep)[0]
        return iter(np.random.choice(kept, size=self.size, replace=True).tolist())

    def __len__(self):
        return self.size


def get_sampler(dataset, alpha=0.5):
    return HeadSubsamplingSampler(dataset.labels, alpha)
