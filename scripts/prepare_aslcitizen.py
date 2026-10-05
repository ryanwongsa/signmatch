"""Trim raw ASL-Citizen videos to their signing span with SignRep, as 256x256 clips."""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from dataloaders.data_utils.video import read_video, to_model_input  # noqa: E402
from models.signrep_active import Model  # noqa: E402

ACTIVE_WINDOW = 16
# bf16 scores shift with the batch size, enough to move a span edge: 4 reproduces the scored clips.
WINDOW_BATCH = 4
THRESHOLD = 0.5
CLIP_SIZE = (256, 256)
DECODE_WORKERS = 8


def kept_span(window_scores, num_frames):
    """(first, last) frame to keep, inclusive, from one active score per window at stride 1."""
    total = torch.zeros(num_frames)
    count = torch.zeros(num_frames)
    for start, score in enumerate(window_scores):
        total[start : start + ACTIVE_WINDOW] += score
        count[start : start + ACTIVE_WINDOW] += 1
    active = torch.nonzero(total / count > THRESHOLD).flatten()
    if len(active) == 0:
        return 0, num_frames - 1
    return int(active[0]), int(active[-1])


def window_scores(model, frames, dtype=torch.bfloat16, batch_size=WINDOW_BATCH):
    if len(frames) < ACTIVE_WINDOW:
        return torch.empty(0)
    windows = frames.cuda().unfold(0, ACTIVE_WINDOW, 1).permute(0, 1, 4, 2, 3)
    scores = []
    autocast = torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32)
    with torch.no_grad(), autocast:
        for batch in windows.split(batch_size):
            logits = model(batch.contiguous())["active_logits"]
            scores.append(logits.max(dim=-1).values.float().cpu())
    return torch.cat(scores)


def load_signrep(path):
    state = torch.load(path, map_location="cpu")["model"]
    state = {key: value for key, value in state.items() if not key.startswith("cls_head.")}
    model = Model()
    model.load_state_dict(state, strict=True)
    return model


class RawVideos(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        frames, fps = read_video(self.paths[index])
        clip = [cv2.resize(f, CLIP_SIZE, interpolation=cv2.INTER_AREA) for f in frames]
        return {
            "name": self.paths[index].name,
            "fps": fps,
            "frames": to_model_input(frames),
            "clip": torch.from_numpy(np.stack(clip)),
        }


def write_clip(frames, fps, path):
    # A temporary name until complete, so an interrupted write is never mistaken for a clip.
    partial = path.with_name(f".{path.stem}.{os.getpid()}.partial.mp4")
    writer = cv2.VideoWriter(str(partial), cv2.VideoWriter_fourcc(*"mp4v"), fps, CLIP_SIZE)
    for frame in frames:
        writer.write(frame)
    writer.release()
    partial.rename(path)


def shard(value):
    index, count = (int(part) for part in value.split("/"))
    return index, count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="directory of raw .mp4 videos")
    parser.add_argument("--signrep", type=Path, required=True, help="SignRep checkpoint")
    parser.add_argument("--out", type=Path, required=True, help="directory for the clips")
    parser.add_argument("--gpu", type=int, help="the one GPU to use (default: the first visible)")
    parser.add_argument(
        "--shard", type=shard, default=(0, 1), help="i/n: process every n-th video from the i-th"
    )
    args = parser.parse_args(argv)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    index, count = args.shard
    videos = sorted(args.raw.glob("*.mp4"))[index::count]
    todo = [video for video in videos if not (args.out / video.name).exists()]
    print(f"shard {index}/{count}: {len(videos)} videos, {len(todo)} to trim")

    model = load_signrep(args.signrep).cuda().eval()
    loader = torch.utils.data.DataLoader(
        RawVideos(todo), batch_size=None, num_workers=DECODE_WORKERS
    )
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    for done, item in enumerate(loader, start=1):
        first, last = kept_span(window_scores(model, item["frames"]), len(item["frames"]))
        write_clip(item["clip"][first : last + 1].numpy(), item["fps"], args.out / item["name"])
        if done % 500 == 0 or done == len(todo):
            minutes = (time.monotonic() - start) / 60
            print(f"{done}/{len(todo)} trimmed in {minutes:.1f} min", flush=True)


if __name__ == "__main__":
    main()
