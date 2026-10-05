import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from dataloaders.data_utils.lmdb_frames import LMDBFrames
from gatle_ignite import build_dataloader

CLIP_LENGTH = 32


def clip_indices(start, end, num_frames):
    if start == end:
        end = start + 1
    if end > num_frames:
        end = num_frames - 1
    if end < start:
        start = end - 1
    first = max((start + end) // 2 - CLIP_LENGTH // 2, 0)
    first = min(first, num_frames - CLIP_LENGTH)
    return np.arange(first, first + CLIP_LENGTH)


class ContinuousClips(Dataset):
    def __init__(self, spottings, lmdbs, transform):
        self.rows = spottings.to_dict(orient="records")
        self.lmdbs = {name: LMDBFrames(root) for name, root in lmdbs.items()}
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        video = self.lmdbs[row["lmdb"]]
        num_frames = video.num_frames(row["video_id"])
        indices = clip_indices(row["start_frame"], row["end_frame"], num_frames)
        frames = self.transform(video.frames(row["video_id"], indices), train=False)
        return {"clip": frames.permute(1, 0, 2, 3), "label": row["label"]}


def get_ds(ds_params, transform=None):
    spottings = pd.read_csv(ds_params["spottings"])
    dataset = ContinuousClips(spottings, ds_params["lmdbs"], transform)
    return build_dataloader(dataset, ds_params), {"length": len(dataset)}
