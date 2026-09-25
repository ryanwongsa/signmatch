from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import torch

from signmatch import SignMatch, iter_continuous_clips, load_dictionary_video

DEFAULT_CKPT = Path(__file__).resolve().parent.parent / "checkpoints" / "signmatch_rgb.pt"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("branch", choices=["dictionary", "continuous"])
    p.add_argument("videos", nargs="*", type=Path)
    p.add_argument("-o", "--out", type=Path, required=True, help="output .pt file")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--frame-stride",
        type=int,
        default=None,
        help="keep every Nth frame; defaults to 2 for dictionary, 1 for continuous. "
        "Both defaults assume a 25fps source.",
    )
    p.add_argument("--chunk-size", type=int, default=16, help="clips per backbone forward")
    p.add_argument(
        "--window-stride",
        type=int,
        default=16,
        help="continuous only: frame step between windows",
    )
    p.add_argument(
        "--block-frames",
        type=int,
        default=1024,
        help="continuous only: frames decoded at a time, so long videos stream",
    )
    args = p.parse_args()

    if args.frame_stride is None:
        args.frame_stride = 2 if args.branch == "dictionary" else 1

    if not args.videos:
        p.error("give video paths")

    model = SignMatch.from_checkpoint(args.checkpoint, device=args.device)
    print(f"loaded {args.checkpoint} on {args.device}")

    records = []
    for path in args.videos:
        if args.branch == "dictionary":
            clips = load_dictionary_video(path, frame_stride=args.frame_stride)
            lengths = torch.tensor([clips.shape[0]])
            emb = model.encode_dictionary(
                clips.to(args.device), lengths, chunk_size=args.chunk_size
            )[0]
            records.append({"path": str(path), "embedding": emb.cpu()})
            print(f"  {path.name}: {clips.shape[0]} clips -> 1 embedding")
        else:
            embs, win_spans = [], []
            for clips, block_spans in iter_continuous_clips(
                path,
                frame_stride=args.frame_stride,
                stride=args.window_stride,
                block_frames=args.block_frames,
            ):
                embs.append(
                    model.encode_continuous(
                        clips.to(args.device), chunk_size=args.chunk_size
                    ).cpu()
                )
                win_spans.extend(block_spans)
                print(f"  {path.name}: {len(win_spans)} windows")
            emb = torch.cat(embs)
            records.append({"path": str(path), "embedding": emb, "spans": win_spans})
            print(f"  {path.name}: done -> {tuple(emb.shape)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"branch": args.branch, "records": records}, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
