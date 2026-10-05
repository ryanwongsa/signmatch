import ignite.distributed as idist
import torch
import torch.nn.functional as F
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced

from gatle_ignite import get_value
from metrics.retrieval import summarise


class DictionaryRanking(Metric):
    def __init__(self, engine_type, **kwargs):
        self.engine_type = engine_type
        super().__init__(**kwargs)

    @reinit__is_reduced
    def reset(self):
        self.ranks = []
        self.labels = []

    @reinit__is_reduced
    def update(self, output):
        clips, labels, dictionary = output
        similarity = (
            F.normalize(clips.float(), dim=1) @ F.normalize(dictionary["embeddings"], dim=1).T
        )
        index = {gloss: i for i, gloss in enumerate(dictionary["ids"])}
        targets = torch.tensor([index[label] for label in labels], device=similarity.device)
        own = similarity.gather(1, targets[:, None])
        self.ranks += (similarity > own).sum(dim=1).tolist()
        self.labels += labels

    def compute(self):
        if idist.get_world_size() > 1:
            raise RuntimeError(
                "dictionary ranking averages over every clip's label, so it must see every clip "
                "in one process; evaluate on a single GPU."
            )
        codes = {label: code for code, label in enumerate(dict.fromkeys(self.labels))}
        label_codes = torch.tensor([codes[label] for label in self.labels])
        summary = summarise(torch.tensor(self.ranks), label_codes, len(codes))
        metrics = {**summary["average_metrics"], **summary["class_metrics"]}
        return {f"{self.engine_type}/{name}": value for name, value in metrics.items()}


def get_metric(engine_type, info, src_name="clip_embeddings"):
    def output_transform(output):
        targets = output["target"]["targets"]
        return get_value(output["y_pred"], src_name), targets["label"], targets["dictionary"]

    return DictionaryRanking(
        engine_type, output_transform=output_transform, metrics_result_mode="flatten"
    )
