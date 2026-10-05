"""Score a weights file on ASL-Citizen retrieval; write the metrics and embeddings to --out."""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from configs.aslcitizen import get_config  # noqa: E402
from gatle_ignite import load_weights  # noqa: E402
from models.signmatch import dictionary_head_of  # noqa: E402
from trainer.aslcitizen_trainer import Trainer  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", type=Path, required=True, help="bare or {'model': ...} state dict"
    )
    parser.add_argument("--clips", type=Path, required=True, help="directory of trimmed clips")
    parser.add_argument("--out", type=Path, required=True, help="results directory")
    parser.add_argument("--gpu", type=int, help="the one GPU to use (default: the first visible)")
    args = parser.parse_args(argv)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    args.out.mkdir(parents=True, exist_ok=True)
    weights = torch.load(args.weights, map_location="cpu")
    cfg = get_config()
    cfg.model_params.dictionary_head = dictionary_head_of(weights.get("model", weights))
    cfg.test_ds_params.clips_dir = str(args.clips)
    cfg.tester_metrics.retrieval.params.out_dir = str(args.out)
    cfg.run = True

    # Built directly rather than launched, which would start one process per visible GPU.
    trainer = Trainer(0, cfg).setup()
    load_weights(weights, trainer.model, strict=True)
    start = time.monotonic()
    trainer.evaluate(load_checkpoint=False)
    print(f"scored in {(time.monotonic() - start) / 60:.1f} min; results in {args.out}")


if __name__ == "__main__":
    main()
