from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent)) 

import argparse

import torch

from signmatch import match


def _stack(bundle: dict) -> tuple[torch.Tensor, list[str], list]:
    rows, labels, spans = [], [], []
    for rec in bundle["records"]:
        emb = rec["embedding"]
        if emb.ndim == 1:
            emb = emb.unsqueeze(0)
        rows.append(emb)
        name = Path(rec["path"]).stem
        if "spans" in rec:
            labels.extend([name] * emb.shape[0])
            spans.extend(rec["spans"])
        else:
            labels.append(name)
            spans.append(None)
    return torch.cat(rows, dim=0), labels, spans


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--queries", type=Path, required=True)
    p.add_argument("--dictionary", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="only report matches at or above this cosine score",
    )
    args = p.parse_args()

    q, q_labels, q_spans = _stack(torch.load(args.queries, weights_only=False))
    d, d_labels, _ = _stack(torch.load(args.dictionary, weights_only=False))
    print(f"{q.shape[0]} queries x {d.shape[0]} dictionary entries\n")

    scores = match(q, d)
    k = min(args.top_k, d.shape[0])
    top_scores, top_idx = scores.topk(k, dim=-1)

    for i, label in enumerate(q_labels):
        if args.threshold is not None and top_scores[i, 0].item() < args.threshold:
            continue
        where = f" frames {q_spans[i][0]}-{q_spans[i][1]}" if q_spans[i] else ""
        print(f"{label}{where}")
        for score, idx in zip(top_scores[i].tolist(), top_idx[i].tolist()):
            print(f"    {score:6.3f}  {d_labels[idx]}")


if __name__ == "__main__":
    main()
