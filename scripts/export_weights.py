"""Write a weights file from a training checkpoint: the inference tensors only, values unchanged."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from models.signmatch import Model, dictionary_head_of  # noqa: E402


def export(checkpoint):
    model = Model(dictionary_head=dictionary_head_of(checkpoint["model"]))
    weights = {key: checkpoint["model"][key] for key in model.state_dict()}
    model.load_state_dict(weights, strict=True)
    return weights


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="training checkpoint")
    parser.add_argument("--out", type=Path, required=True, help="weights file")
    args = parser.parse_args(argv)

    weights = export(torch.load(args.src, map_location="cpu"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, args.out)
    print(f"wrote {len(weights)} tensors to {args.out}")


if __name__ == "__main__":
    main()
