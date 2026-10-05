import torch.nn as nn
import torch.nn.functional as F

from gatle_ignite import get_value


class Loss(nn.Module):
    def __init__(
        self,
        temperature,
        src_name="logits",
        tgt_name=("targets", "label"),
        similarity_name="sim_confusion_matrix",
    ):
        super().__init__()
        self.temperature = temperature
        self.src_name = src_name
        self.tgt_name = tgt_name
        self.similarity_name = similarity_name

    def forward(self, y_pred, target, iteration=None):
        logits = get_value(y_pred, self.src_name)
        confusion = F.softmax(get_value(y_pred, self.similarity_name) / self.temperature, dim=1)
        q = confusion[get_value(target, self.tgt_name)].detach()
        log_p = F.log_softmax(logits / self.temperature, dim=1)
        return -(q * log_p).sum(dim=1).mean()
