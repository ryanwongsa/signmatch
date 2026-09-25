from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

IMAGE_SIZE = 224
WINDOW = 32
STRIDE = 16
MAX_DICT_FRAMES = 80
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def read_video_frames(path: str | Path, frame_stride: int = 1) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames[::frame_stride]


def preprocess_frames(frames: list[np.ndarray]) -> torch.Tensor:
    if not frames:
        raise ValueError("no frames to preprocess")
    resized = np.stack([f if f.shape[:2] == (IMAGE_SIZE, IMAGE_SIZE) else cv2.resize(f, (IMAGE_SIZE, IMAGE_SIZE)) for f in frames])
    x = torch.from_numpy(resized).float().div_(255.0)
    x = (x - torch.tensor(MEAN)) / torch.tensor(STD)
    return x.permute(0, 3, 1, 2).contiguous()


def _fit_to_window_grid(frames: torch.Tensor) -> torch.Tensor:
    n = frames.shape[0]
    target = WINDOW if n <= WINDOW else WINDOW + int(np.ceil((n - WINDOW) / STRIDE)) * STRIDE
    if n < target:
        left = (target - n) // 2
        right = target - n - left
        frames = torch.cat(
            [frames[0:1].repeat(left, 1, 1, 1), frames, frames[-1:].repeat(right, 1, 1, 1)]
        )
    elif n > target:
        frames = frames[np.round(np.linspace(0, n - 1, target)).astype(int)]
    return frames


def dictionary_clips(frames: torch.Tensor) -> torch.Tensor:
    n = frames.shape[0]
    if n > MAX_DICT_FRAMES:
        frames = frames[np.round(np.linspace(0, n - 1, MAX_DICT_FRAMES)).astype(int)]
    windows = _fit_to_window_grid(frames).unfold(0, WINDOW, STRIDE)
    return windows.permute(0, 1, 4, 2, 3).contiguous()


def continuous_clips(
    frames: torch.Tensor, stride: int = STRIDE
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    n = frames.shape[0]
    if n < WINDOW:
        return dictionary_clips(frames), [(0, n)]
    starts = range(0, n - WINDOW + 1, stride)
    clips = torch.stack([frames[s : s + WINDOW] for s in starts])
    return clips.permute(0, 2, 1, 3, 4).contiguous(), [(s, s + WINDOW) for s in starts]


def load_dictionary_video(
    path: str | Path,
    first_active: int | None = None,
    last_active: int | None = None,
    frame_stride: int = 2,
) -> torch.Tensor:
    frames = read_video_frames(path)
    if first_active is not None and last_active is not None:
        frames = frames[first_active : last_active + 1]
    return dictionary_clips(preprocess_frames(frames[::frame_stride]))


def load_continuous_video(
    path: str | Path, frame_stride: int = 1, stride: int = STRIDE
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    return continuous_clips(
        preprocess_frames(read_video_frames(path, frame_stride)), stride=stride
    )


def iter_continuous_clips(
    path: str | Path,
    frame_stride: int = 1,
    stride: int = STRIDE,
    block_frames: int = 1024,
):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {path}")

    buf: list[np.ndarray] = []
    base = 0 
    kept = 0
    seen = 0

    def flush(final: bool):
        nonlocal buf, base, kept
        n = len(buf)
        if n < WINDOW:
            return None
        k = (n - WINDOW) // stride + 1
        frames = preprocess_frames(buf)
        windows = frames.unfold(0, WINDOW, stride)[:k]
        spans = [(base + i * stride, base + i * stride + WINDOW) for i in range(k)]
        drop = k * stride if not final else n
        buf = buf[drop:]
        base += drop
        kept += k
        return windows.permute(0, 1, 4, 2, 3).contiguous(), spans

    try:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % frame_stride == 0:
                buf.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (IMAGE_SIZE, IMAGE_SIZE)))
                seen += 1
                if len(buf) >= block_frames:
                    out = flush(final=False)
                    if out is not None:
                        yield out
            i += 1
    finally:
        cap.release()

    out = flush(final=True)
    if out is not None:
        yield out
    elif kept == 0 and seen:
        yield dictionary_clips(preprocess_frames(buf)), [(0, seen)]
