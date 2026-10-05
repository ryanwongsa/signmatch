import math

import ignite.distributed as idist
import pandas as pd
import torch
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced

from dataloaders.data_utils.classes import class_index
from gatle_ignite import get_value
from metrics.retrieval import summarise


class ClassRanking(Metric):
    def __init__(self, engine_type, index, mask, **kwargs):
        self.engine_type = engine_type
        self.index = index
        self.mask = mask
        super().__init__(**kwargs)

    @reinit__is_reduced
    def reset(self):
        self.ranks = []
        self.labels = []

    @reinit__is_reduced
    def update(self, output):
        logits, labels = output
        logits = logits.float().masked_fill(~self.mask.to(logits.device), -torch.inf)
        for row, label in zip(logits, labels):
            target = self.index.get(label)
            if target is None:
                self.ranks.append(math.inf)
            else:
                self.ranks.append((row > row[target]).sum().item())
        self.labels += labels

    def compute(self):
        if idist.get_world_size() > 1:
            raise RuntimeError(
                "class ranking averages over every clip's label, so it must see every clip in "
                "one process; evaluate on a single GPU."
            )
        codes = {label: code for code, label in enumerate(dict.fromkeys(self.labels))}
        label_codes = torch.tensor([codes[label] for label in self.labels])
        summary = summarise(torch.tensor(self.ranks), label_codes, len(codes))
        metrics = {**summary["average_metrics"], **summary["class_metrics"]}
        return {f"{self.engine_type}/{name}": value for name, value in metrics.items()}


def get_metric(engine_type, info, masked_to, src_name="logits"):
    classes = info["classes"]
    mask = torch.tensor(classes["id"].isin(set(pd.read_csv(masked_to)["id"])).to_numpy())

    def output_transform(output):
        return get_value(output["y_pred"], src_name), output["target"]["targets"]["label"]

    return ClassRanking(
        engine_type,
        class_index(classes),
        mask,
        output_transform=output_transform,
        metrics_result_mode="flatten",
    )
