import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dataloaders.data_utils.lmdb_frames import LMDBFrames
from dataloaders.data_utils.windows import to_windows
from gatle_ignite import build_dataloader

MAX_EXTENSION = 8
MAX_FRAMES = 80


def frame_indices(first, last, num_frames, train):
    """`first` and `last` are inclusive."""
    if train:
        first -= np.random.randint(0, min(first, MAX_EXTENSION) + 1)
        last += np.random.randint(0, min(num_frames - 1 - last, MAX_EXTENSION) + 1)
    indices = np.arange(first, last + 1, 2)
    if len(indices) > MAX_FRAMES:
        indices = indices[np.round(np.linspace(0, len(indices) - 1, MAX_FRAMES)).astype(int)]
    return indices


class DictionaryVideos(Dataset):
    def __init__(self, manifest, lmdbs, transform, train):
        self.rows = manifest.to_dict(orient="records")
        self.lmdbs = {name: LMDBFrames(root) for name, root in lmdbs.items()}
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        indices = frame_indices(
            row["first_active"], row["last_active"], row["num_frames"], self.train
        )
        frames = self.lmdbs[row["lmdb"]].frames(row["video_name"], indices)
        windows = to_windows(self.transform(frames, train=self.train), random_split=self.train)
        return {"windows": windows, "length": len(windows), "label": index, "id": row["id"]}


def collate(items):
    return {
        "windows": torch.cat([item["windows"] for item in items]),
        "lengths": torch.tensor([item["length"] for item in items]),
        "label": torch.tensor([item["label"] for item in items]),
        "id": [item["id"] for item in items],
    }


def get_ds(ds_params, transform=None):
    manifest = pd.read_csv(ds_params["manifest"])
    dataset = DictionaryVideos(manifest, ds_params["lmdbs"], transform, ds_params["train"])
    loader = build_dataloader(dataset, {**ds_params, "collate_fn": collate})
    return loader, {"length": len(dataset)}
