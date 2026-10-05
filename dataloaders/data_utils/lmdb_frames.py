import pickle

import cv2
import lmdb
import numpy as np


class LMDBFrames:
    def __init__(self, root):
        self.root = root
        self.txn = None

    def _get(self, key):
        # Opened on first use, so each DataLoader worker opens its own handle.
        if self.txn is None:
            env = lmdb.open(self.root, readonly=True, lock=False, readahead=False, meminit=False)
            self.txn = env.begin(write=False)
        value = self.txn.get(key.encode("ascii"))
        if value is None:
            raise KeyError(f"{key} not in {self.root}")
        return value

    def num_frames(self, video):
        return pickle.loads(self._get(f"{video}_details"))["num_frames"]

    def frames(self, video, indices):
        """-> RGB uint8 frames, in the order of `indices`."""
        frames = []
        for i in indices:
            jpeg = np.frombuffer(self._get(f"{video}_{i}"), np.uint8)
            frames.append(cv2.cvtColor(cv2.imdecode(jpeg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
        return frames
