import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from dataloaders.data_utils.classes import class_index, class_list
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
    centre = np.random.normal((start + end) / 2, (end - start) / 6)
    centre = int(np.round(np.clip(centre, start, end - 1)))
    first = max(centre - CLIP_LENGTH // 2, 0)
    first = min(first, num_frames - CLIP_LENGTH)
    return np.arange(first, first + CLIP_LENGTH)


class SpottingClips(Dataset):
    def __init__(self, spottings, classes, lmdbs, transform):
        self.rows = spottings.to_dict(orient="records")
        index = class_index(classes)
        self.labels = np.array([index[row["label"]] for row in self.rows])
        self.lmdbs = {name: LMDBFrames(root) for name, root in lmdbs.items()}
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        video = self.lmdbs[row["lmdb"]]
        num_frames = video.num_frames(row["video_id"])
        indices = clip_indices(row["start_frame"], row["end_frame"], num_frames)
        frames = self.transform(video.frames(row["video_id"], indices), train=True)
        return {"clip": frames.permute(1, 0, 2, 3), "label": self.labels[index]}


def get_ds(ds_params, transform=None):
    spottings = pd.read_csv(ds_params["spottings"])
    classes = class_list(pd.read_csv(ds_params["dictionary"]), spottings)
    dataset = SpottingClips(spottings, classes, ds_params["lmdbs"], transform)
    return build_dataloader(dataset, ds_params), {"length": len(dataset), "classes": classes}
