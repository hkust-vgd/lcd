import os
import h5py
import numpy as np
import datetime
from PIL import Image
from itertools import chain


cloud_size = 1024
image_size = 64
radius = 0.15
batch_size = 10000


def tostring(s):
    s = [chr(x) for x in s]
    return "".join(s)


def extract_point_cloud(depth, color, bbox, origin, K):
    cloud = []
    for v in range(bbox[0, 1], bbox[1, 1]):
        for u in range(bbox[0, 0], bbox[1, 0]):
            z = depth[v - bbox[0, 1], u - bbox[0, 0]]
            x = (u - K[0, 2]) * z / K[0, 0]
            y = (v - K[1, 2]) * z / K[1, 1]
            r = color[v - bbox[0, 1], u - bbox[0, 0], 0] / 255.0
            g = color[v - bbox[0, 1], u - bbox[0, 0], 1] / 255.0
            b = color[v - bbox[0, 1], u - bbox[0, 0], 2] / 255.0
            if z <= 0.0:
                continue
            if abs(x - origin[0]) >= radius:
                continue
            if abs(y - origin[1]) >= radius:
                continue
            if abs(z - origin[2]) >= radius:
                continue
            cloud.append([x, y, z, r, g, b])
    # Subsample point cloud
    cloud = np.array(cloud, dtype=np.float32)
    indices = np.random.choice(cloud.shape[0], cloud_size, replace=True)
    cloud = cloud[indices, :]
    cloud[:, 0:3] = (cloud[:, 0:3] - origin) / radius
    return cloud


def extract_color_patch(color):
    image = Image.fromarray(color)
    image = image.resize((image_size, image_size))
    image = np.array(image, dtype=np.float32) / 255.0
    return image


def save_batch_h5(fname, batch):
    patches = list(chain(*batch))
    points = np.stack([patch["cloud"] for patch in patches])
    images = np.stack([patch["color"] for patch in patches])
    fp = h5py.File(fname, "w")
    fp.create_dataset("points", data=points, compression="gzip")
    fp.create_dataset("images", data=images, compression="gzip")
    fp.close()


fname = os.path.join("data", "3dmatch", "test-set.mat")
print("> Loading test data from {}...".format(fname))
fin = h5py.File(fname)
dataset = fin["data"]
patch0 = dataset[0]
patch1 = dataset[1]

batch = []
dataset_size = len(patch0)
size = 0
while size < dataset_size:
    patches = []
    # Extract first patch
    path = tostring(fin[patch0[size]]["framePath"][()])
    depth = np.transpose(fin[patch0[size]]["depthPatch"][()])
    color = np.transpose(fin[patch0[size]]["colorPatch"][()], [2, 1, 0])
    bbox = fin[patch0[size]]["bboxRangePixels"][()].astype(np.int32)
    K = np.transpose(fin[patch0[size]]["camK"][()])
    p = fin[patch0[size]]["camCoords"][()][0]

    patch = {}
    patch["cloud"] = extract_point_cloud(depth, color, bbox, p, K)
    patch["color"] = extract_color_patch(color)
    patches += [patch]

    # Extract second patch
    path = tostring(fin[patch1[size]]["framePath"][()])
    depth = np.transpose(fin[patch1[size]]["depthPatch"][()])
    color = np.transpose(fin[patch1[size]]["colorPatch"][()], [2, 1, 0])
    bbox = fin[patch1[size]]["bboxRangePixels"][()].astype(np.int32)
    K = np.transpose(fin[patch1[size]]["camK"][()])
    p = fin[patch1[size]]["camCoords"][()][0]

    patch = {}
    patch["cloud"] = extract_point_cloud(depth, color, bbox, p, K)
    patch["color"] = extract_color_patch(color)
    patches += [patch]

    size += 1
    batch += [patches]
    print("Extracted patches [{}/{}]".format(size, dataset_size))

    if len(batch) == batch_size:
        i = size // batch_size
        fname = "data/3dmatch/h5/test/{:04d}.h5".format(i)
        print("> Saving batch to {}...".format(fname))
        save_batch_h5(fname, batch)
        batch = []
