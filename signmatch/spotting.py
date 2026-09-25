from __future__ import annotations

import torch

FPS = 25.0


def timecode(frame: int, fps: float = FPS) -> str:
    s = frame / fps
    return f"{int(s // 60):02d}:{s % 60:05.2f}"


def pick(scores: torch.Tensor, spans, top_k: int, per_gloss: int, gap: int, floor: float = -1.0):
    flat = scores.flatten()
    n_gloss = scores.shape[1]
    
    depth = min(flat.numel(), max(top_k * 200, 10_000))
    vals, order = torch.topk(flat, depth)
    chosen, counts = [], {}
    for score, k in zip(vals.tolist(), order.tolist()):
        if score < floor:
            break
        w, g = divmod(k, n_gloss)
        if counts.get(g, 0) >= per_gloss:
            continue
        if any(g == cg and abs(spans[w][0] - spans[cw][0]) < gap for cw, cg, _ in chosen):
            continue
        chosen.append((w, g, score))
        counts[g] = counts.get(g, 0) + 1
        if len(chosen) == top_k:
            break
    return chosen
