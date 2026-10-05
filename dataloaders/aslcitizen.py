from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from dataloaders.data_utils.video import read_video, to_model_input
from dataloaders.data_utils.windows import to_windows
from gatle_ignite import build_dataloader

SPLITS = ("train", "val", "test")


def read_splits(splits_dir):
    rows = []
    for split in SPLITS:
        table = pd.read_csv(Path(splits_dir) / f"{split}.csv")
        rows += [(video, gloss, split) for video, gloss in zip(table["Video file"], table["Gloss"])]
    return rows


class ASLCitizen(Dataset):
    def __init__(self, rows, clips_dir):
        self.rows = rows
        self.clips_dir = Path(clips_dir)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        video_id, gloss, split = self.rows[index]
        path = self.clips_dir / video_id
        frames, _ = read_video(path)
        if not frames:
            raise FileNotFoundError(f"no frames decoded from {path}: missing or unreadable")
        windows = to_windows(to_model_input(frames[::2]))
        return {
            "video_id": video_id,
            "gloss": gloss,
            "split": split,
            "windows": windows,
            "length": len(windows),
        }


def collate(items):
    return {
        "windows": torch.cat([item["windows"] for item in items]),
        "lengths": torch.tensor([item["length"] for item in items]),
        "video_id": [item["video_id"] for item in items],
        "gloss": [item["gloss"] for item in items],
        "split": [item["split"] for item in items],
    }


def get_ds(ds_params, transform=None):
    dataset = ASLCitizen(read_splits(ds_params["splits_dir"]), ds_params["clips_dir"])
    loader = build_dataloader(dataset, {**ds_params, "collate_fn": collate})
    return loader, {"length": len(dataset)}
