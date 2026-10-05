import json
import math
import pickle
from pathlib import Path

import ignite.distributed as idist
import torch
import torch.nn.functional as F
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced

from gatle_ignite import get_value

GALLERY_SPLIT = "train"
QUERY_SPLITS = ("val", "test")
TOP_K = (1, 5, 10, 20)
QUERY_CHUNK = 1024


def class_ranks(queries, query_classes, gallery, gallery_classes, num_classes):
    ranks = []
    for chunk, labels in zip(queries.split(QUERY_CHUNK), query_classes.split(QUERY_CHUNK)):
        similarity = chunk @ gallery.T
        scores = torch.full((len(chunk), num_classes), -math.inf)
        scores = scores.scatter_reduce(
            1, gallery_classes.expand(len(chunk), -1), similarity, reduce="amax"
        )
        true_scores = scores.gather(1, labels[:, None])
        ranks.append((scores > true_scores).sum(dim=1))
    return torch.cat(ranks)


def summarise(ranks, query_classes, num_classes):
    """`ranks` are 0-based; DCG = 1 / log2(rank + 2)."""
    ranks = ranks.double()
    per_query = {"DCG": 1 / torch.log2(ranks + 2), "MRR": 1 / (ranks + 1)}
    for k in TOP_K:
        per_query[f"Top{k}"] = (ranks < k).double()

    counts = torch.bincount(query_classes, minlength=num_classes).double()
    present = counts > 0
    average_metrics, class_metrics = {}, {}
    for name, values in per_query.items():
        average_metrics[name] = values.mean().item()
        sums = torch.zeros(num_classes, dtype=torch.float64).index_add_(0, query_classes, values)
        class_metrics[f"C{name}"] = (sums[present] / counts[present]).mean().item()
    return {"average_metrics": average_metrics, "class_metrics": class_metrics}


def retrieval_results(embeddings, glosses, splits):
    embeddings = F.normalize(embeddings, dim=1)
    gallery = [i for i, split in enumerate(splits) if split == GALLERY_SPLIT]
    classes = sorted({glosses[i] for i in gallery})
    class_index = {gloss: c for c, gloss in enumerate(classes)}
    gallery_classes = torch.tensor([class_index[glosses[i]] for i in gallery])

    results = {}
    for query_split in QUERY_SPLITS:
        queries = [
            i
            for i, split in enumerate(splits)
            if split == query_split and glosses[i] in class_index
        ]
        if not queries:
            continue
        query_classes = torch.tensor([class_index[glosses[i]] for i in queries])
        ranks = class_ranks(
            embeddings[queries], query_classes, embeddings[gallery], gallery_classes, len(classes)
        )
        results[query_split] = {
            "num_train_instances": len(gallery),
            "num_query_instances": len(queries),
            "num_classes_in_train": len(classes),
            **summarise(ranks, query_classes, len(classes)),
        }
    return results


class Retrieval(Metric):
    def __init__(self, out_dir=None, **kwargs):
        self.out_dir = out_dir
        super().__init__(**kwargs)

    @reinit__is_reduced
    def reset(self):
        self.embeddings = []
        self.video_ids = []
        self.glosses = []
        self.splits = []

    @reinit__is_reduced
    def update(self, output):
        embeddings, targets = output
        self.embeddings.append(embeddings.detach().float().cpu())
        self.video_ids += targets["video_id"]
        self.glosses += targets["gloss"]
        self.splits += targets["split"]

    def compute(self):
        if idist.get_world_size() > 1:
            raise RuntimeError(
                "retrieval ranks against the whole gallery, so it must see every video in one "
                "process; evaluate on a single GPU."
            )
        embeddings = torch.cat(self.embeddings)
        results = retrieval_results(embeddings, self.glosses, self.splits)
        if self.out_dir is not None:
            self.save(embeddings, results)

        flat = {}
        for split, result in results.items():
            for name in ("num_train_instances", "num_query_instances", "num_classes_in_train"):
                flat[f"{split}/{name}"] = result[name]
            for name, value in {**result["average_metrics"], **result["class_metrics"]}.items():
                flat[f"{split}/{name}"] = value
        return flat

    def save(self, embeddings, results):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        features = {video_id: row.clone() for video_id, row in zip(self.video_ids, embeddings)}
        with open(out_dir / "aslcitizen_dictionary_features.pkl", "wb") as f:
            pickle.dump(features, f)
        with open(out_dir / "aslcitizen_retrieval_metrics.json", "w") as f:
            json.dump(results, f, indent=2)


def get_metric(engine_type, info, src_name="dictionary_z_features", out_dir=None):
    def output_transform(output):
        return get_value(output["y_pred"], src_name), output["target"]["targets"]

    return Retrieval(
        out_dir=out_dir, output_transform=output_transform, metrics_result_mode="flatten"
    )
