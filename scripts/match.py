"""Rank dictionary entries for each query, from two files scripts/embed.py wrote."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from inference import match  # noqa: E402


def stack(bundle):
    rows, names, spans = [], [], []
    for record in bundle["records"]:
        embedding = record["embedding"]
        if embedding.ndim == 1:
            embedding = embedding[None]
        rows.append(embedding)
        name = Path(record["path"]).stem
        names += [name] * len(embedding)
        spans += record.get("spans", [None])
    return torch.cat(rows), names, spans


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--threshold", type=float, help="only print queries whose best match scores at least this"
    )
    args = parser.parse_args(argv)

    queries, query_names, query_spans = stack(torch.load(args.queries))
    dictionary, dictionary_names, _ = stack(torch.load(args.dictionary))
    print(f"{len(queries)} queries x {len(dictionary)} dictionary entries\n")

    k = min(args.top_k, len(dictionary))
    top_scores, top_entries = match(queries, dictionary).topk(k, dim=1)
    for i, name in enumerate(query_names):
        if args.threshold is not None and top_scores[i, 0].item() < args.threshold:
            continue
        span = query_spans[i]
        print(f"{name} frames {span[0]}-{span[1]}" if span else name)
        for score, entry in zip(top_scores[i].tolist(), top_entries[i].tolist()):
            print(f"    {score:6.3f}  {dictionary_names[entry]}")


if __name__ == "__main__":
    main()
