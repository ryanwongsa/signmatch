import math

import numpy as np
import torch

WINDOW = 32
STRIDE = 16


def to_windows(frames, random_split=False):
    """(T, C, H, W) -> (num_windows, C, 32, H, W), edge frames padding T to the grid."""
    n = len(frames)
    target = WINDOW if n <= WINDOW else WINDOW + math.ceil((n - WINDOW) / STRIDE) * STRIDE
    pad = target - n
    left = np.random.randint(0, pad + 1) if random_split else pad // 2
    padded = torch.cat(
        [frames[:1].repeat(left, 1, 1, 1), frames, frames[-1:].repeat(pad - left, 1, 1, 1)]
    )
    return padded.unfold(0, WINDOW, STRIDE).permute(0, 1, 4, 2, 3)
