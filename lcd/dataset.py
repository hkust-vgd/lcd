import os
import h5py
import glob
import bisect
import numpy as np
import torch.utils.data as data


class CrossTripletDataset(data.Dataset):
    def __init__(self, root, split, cache_size=32):
        self.views = 2
        self.split = split
        self.metadata = {}
        self.cache = {}
        self.cache_size = cache_size
        self.flist = os.path.join(root, split, "*.h5")
        self.flist = sorted(glob.glob(self.flist))
        for fname in self.flist:
            self._add_metadata(fname)

    def __len__(self):
        return sum([s for _, s in self.metadata.items()])

    def __getitem__(self, i):
        for fname, size in self.metadata.items():
            if i < size:
                break
            i -= size
        if fname not in self.cache:
            self._load_data(fname)
        points, images = self.cache[fname]
        return points[i], images[i]

    def _add_metadata(self, fname):
        with h5py.File(fname, "r") as h5:
            assert "points" in h5 and "images" in h5
            assert h5["points"].shape[0] == h5["images"].shape[0]
            size = h5["points"].shape[0]
            self.metadata[fname] = size

    def _load_data(self, fname):
        # Remove a random element from cache
        if len(self.cache) == self.cache_size:
            key = list(self.cache.keys())[0]
            self.cache.pop(key)

        h5 = h5py.File(fname, "r")
        data = (h5["points"], h5["images"])
        self.cache[fname] = data
