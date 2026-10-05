"""Embed sign videos with SignMatch's dictionary or continuous branch into one .pt file."""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from inference import (  # noqa: E402
    embed_continuous,
    embed_dictionary,
    iter_continuous_video,
    load_dictionary_video,
    load_model,
)


def read_manifest(path):
    with open(path) as f:
        return {
            row["video_name"]: (int(row["first_active"]), int(row["last_active"]))
            for row in csv.DictReader(f)
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("branch", choices=["dictionary", "continuous"])
    parser.add_argument("videos", nargs="*", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="dictionary only: a CSV with video_name, first_active and last_active; embeds the "
        "videos it lists, each trimmed to its active span",
    )
    parser.add_argument("--video-dir", type=Path, help="where the manifest's <video_name>.mp4 are")
    parser.add_argument("-o", "--out", type=Path, required=True, help="output .pt file")
    parser.add_argument("--checkpoint", type=Path, required=True, help="SignMatch weights file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--frame-stride", type=int, default=1, help="continuous only: keep every n-th frame"
    )
    parser.add_argument("--chunk-size", type=int, default=16, help="windows per backbone pass")
    parser.add_argument(
        "--window-stride", type=int, default=16, help="continuous only: frames between windows"
    )
    parser.add_argument(
        "--block-frames",
        type=int,
        default=1024,
        help="continuous only: frames decoded at a time, so a long video streams",
    )
    args = parser.parse_args(argv)

    spans = {}
    if args.manifest:
        if args.branch != "dictionary":
            parser.error("--manifest is for the dictionary branch")
        if not args.video_dir:
            parser.error("--manifest needs --video-dir")
        spans = read_manifest(args.manifest)
        args.videos = [args.video_dir / f"{name}.mp4" for name in spans]
    elif not args.videos:
        parser.error("give video paths, or --manifest with --video-dir")

    model = load_model(args.checkpoint, args.device, chunk_size=args.chunk_size)
    print(f"loaded {args.checkpoint} on {args.device}")

    records = []
    for path in args.videos:
        if args.branch == "dictionary":
            first, last = spans.get(path.stem, (None, None))
            windows = load_dictionary_video(path, first, last)
            embedding = embed_dictionary(model, windows).cpu()
            records.append({"path": str(path), "embedding": embedding})
            print(f"{path.name}: {len(windows)} windows -> 1 embedding")
            continue

        embeddings, window_spans = [], []
        blocks = iter_continuous_video(
            path,
            frame_stride=args.frame_stride,
            stride=args.window_stride,
            block_frames=args.block_frames,
        )
        for windows, block_spans in blocks:
            embeddings.append(embed_continuous(model, windows).cpu())
            window_spans += block_spans
        embedding = torch.cat(embeddings)
        records.append({"path": str(path), "embedding": embedding, "spans": window_spans})
        print(f"{path.name}: {len(window_spans)} windows -> {tuple(embedding.shape)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"branch": args.branch, "records": records}, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
