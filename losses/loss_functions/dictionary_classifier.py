import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from gatle_ignite import get_value


def temperature(num_classes, p):
    return 1.0 / (math.log(num_classes - 1) - math.log(1.0 / p - 1.0))


class Loss(nn.Module):
    def __init__(
        self,
        temperature,
        src_name="dictionary_z_features",
        tgt_name=("targets", "label"),
        prototypes_name="norm_A",
        similarity_name="sim_confusion_matrix",
    ):
        super().__init__()
        self.temperature = temperature
        self.src_name = src_name
        self.tgt_name = tgt_name
        self.prototypes_name = prototypes_name
        self.similarity_name = similarity_name

    def forward(self, y_pred, target, iteration=None):
        embeddings = get_value(y_pred, self.src_name)
        # In the dtypes the model produced them, not autocast's.
        with torch.autocast(embeddings.device.type, enabled=False):
            similarity = get_value(y_pred, self.similarity_name)
            confusion = F.softmax(similarity / self.temperature, dim=1)
            q = confusion[get_value(target, self.tgt_name)].detach()
            logits = embeddings @ get_value(y_pred, self.prototypes_name).T
            log_p = F.log_softmax(logits / self.temperature, dim=1)
            return -(q * log_p).sum(dim=1).mean()
