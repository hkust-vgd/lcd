import os
import h5py
import glob
import bisect
import numpy as np
import torch.utils.data as data


class CrossTripletDataset(data.Dataset):
    def __init__(self, root, split):
        self.views = 2
        self.split = split
        self.flist = os.path.join(root, 'h5', split, '*.h5')
        self.flist = sorted(glob.glob(self.flist))
        self.flist = [h5py.File(fname, 'r') for fname in self.flist]
        self.points = np.concatenate([f['points'][:] for f in self.flist])
        self.images = np.concatenate([f['images'][:] for f in self.flist])
        self.size = self.points.shape[0] // self.views

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        start = i * self.views
        end = i * self.views + self.views
        points = self.points[start:end]
        images = self.images[start:end]
        return points[0], images[1], points[1], images[0]
