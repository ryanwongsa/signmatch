"""Embed dictionary and continuous sign videos with SignMatch, match them, and spot signs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from dataloaders.data_utils.video import iter_frames, read_video, to_model_input  # noqa: E402
from dataloaders.data_utils.windows import STRIDE, WINDOW, to_windows  # noqa: E402
from dataloaders.dictionary import frame_indices  # noqa: E402
from models.signmatch import Model, dictionary_head_of  # noqa: E402

FPS = 25.0


def load_model(weights, device="cpu", chunk_size=16):
    """A weights file (bare or under `model`), loaded strictly: not a training checkpoint."""
    state = torch.load(weights, map_location="cpu")
    state = state.get("model", state)
    model = Model(chunk_size=chunk_size, dictionary_head=dictionary_head_of(state))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_dictionary_video(path, first_active=None, last_active=None):
    """(windows, 3, 32, 224, 224): every second frame of the inclusive span, at most 80."""
    frames, _ = read_video(path)
    if not frames:
        raise FileNotFoundError(f"no frames decoded from {path}")
    first = 0 if first_active is None else first_active
    last = len(frames) - 1 if last_active is None else last_active
    indices = frame_indices(first, last, len(frames), train=False)
    return to_windows(to_model_input([frames[i] for i in indices]))


def iter_continuous_video(path, frame_stride=1, stride=STRIDE, block_frames=1024):
    """Yield (windows, spans) blocks; a span is a window's source frames, (start, end exclusive)."""
    buffer = []
    first = 0
    # Kept frames still to drop before the next window starts, when `stride` reaches past the block.
    skip = 0
    for i, frame in enumerate(iter_frames(path)):
        if i % frame_stride:
            continue
        if skip:
            skip -= 1
            continue
        buffer.append(frame)
        if len(buffer) >= max(block_frames, WINDOW):
            count = (len(buffer) - WINDOW) // stride + 1
            yield _windows(buffer, first, count, stride, frame_stride)
            first += count * stride
            skip = max(count * stride - len(buffer), 0)
            buffer = buffer[count * stride :]
    if len(buffer) >= WINDOW:
        count = (len(buffer) - WINDOW) // stride + 1
        yield _windows(buffer, first, count, stride, frame_stride)
    elif buffer and first == 0:
        yield to_windows(to_model_input(buffer)), [(0, (len(buffer) - 1) * frame_stride + 1)]


def _windows(frames, first, count, stride, frame_stride):
    windows = to_model_input(frames).unfold(0, WINDOW, stride)[:count].permute(0, 1, 4, 2, 3)
    starts = [first + k * stride for k in range(count)]
    spans = [(start * frame_stride, (start + WINDOW - 1) * frame_stride + 1) for start in starts]
    return windows.contiguous(), spans


@torch.no_grad()
def embed_dictionary(model, windows):
    """One dictionary video's windows -> its (512,) embedding."""
    device = next(model.parameters()).device
    lengths = torch.tensor([len(windows)])
    return model(windows=windows.to(device), lengths=lengths)["dictionary_z_features"][0]


@torch.no_grad()
def embed_continuous(model, windows):
    """Continuous windows -> (num_windows, 512) embeddings."""
    device = next(model.parameters()).device
    return model.embed_clips(windows.to(device))


def match(queries, dictionary):
    """Cosine similarity, (num_queries, num_dictionary), of L2-normalised embeddings."""
    return queries @ dictionary.T


def spot(scores, spans, top_k, per_gloss, gap, floor=-1.0):
    """Greedy best (window, entry, score): `per_gloss` per entry at most, `gap` frames apart."""
    flat = scores.flatten()
    num_entries = scores.shape[1]
    depth = min(flat.numel(), max(top_k * 200, 10_000))
    values, order = torch.topk(flat, depth)
    chosen, counts = [], {}
    for score, k in zip(values.tolist(), order.tolist()):
        if score < floor:
            break
        window, entry = divmod(k, num_entries)
        if counts.get(entry, 0) >= per_gloss:
            continue
        near = [
            taken
            for taken, taken_entry, _ in chosen
            if taken_entry == entry and abs(spans[window][0] - spans[taken][0]) < gap
        ]
        if near:
            continue
        chosen.append((window, entry, score))
        counts[entry] = counts.get(entry, 0) + 1
        if len(chosen) == top_k:
            break
    return chosen


def timecode(frame, fps=FPS):
    """A frame index at `fps` as mm:ss.ss."""
    seconds = frame / fps
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"
