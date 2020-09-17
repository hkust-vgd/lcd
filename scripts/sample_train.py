import os
import sys
import glob
import h5py
import datetime
import numpy as np
from PIL import Image
from itertools import chain


fname = "data/3dmatch/metadata/train.txt"
with open(fname, "r") as fin:
    scenes = [line.strip() for line in fin]
print(scenes)

root = "/run/media/hieu/backup/3dmatch/"
dataset_size = 1000000
batch_size = 10000
min_camera_distance = 1.0
min_match_distance = 0.1
threshold = 0.03
radius = 0.15
cutoff = 4.0
cloud_size = 1024
image_size = 64
num_samples = 5
seed = 42


def get_depth_frames(scene):
    frames = []
    seqs = glob.glob(os.path.join(root, scene, "seq-*/"))
    for seq in seqs:
        frames += sorted(glob.glob(os.path.join(seq, "*.depth.png")))
    return frames


def compute_bounding_box(p):
    return np.array(
        [
            [p[0] - radius, p[1] - radius, p[2] - radius],
            [p[0] - radius, p[1] - radius, p[2] + radius],
            [p[0] - radius, p[1] + radius, p[2] - radius],
            [p[0] - radius, p[1] + radius, p[2] + radius],
            [p[0] + radius, p[1] - radius, p[2] - radius],
            [p[0] + radius, p[1] - radius, p[2] + radius],
            [p[0] + radius, p[1] + radius, p[2] - radius],
            [p[0] + radius, p[1] + radius, p[2] + radius],
        ]
    )


def extract_point_cloud(depth, color, w, h, origin, K):
    cloud = []
    for v in range(h[0], h[1]):
        for u in range(w[0], w[1]):
            z = depth[v, u]
            x = (u - K[0, 2]) * z / K[0, 0]
            y = (v - K[1, 2]) * z / K[1, 1]
            r = color[v, u, 0] / 255.0
            g = color[v, u, 1] / 255.0
            b = color[v, u, 2] / 255.0
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


def extract_color_patch(color, w, h):
    image = Image.fromarray(color[h[0] : h[1], w[0] : w[1]])
    image = image.resize((image_size, image_size))
    image = np.array(image, dtype=np.float32) / 255.0
    return image


def sample_matching_pairs(scene):
    patches = []
    frames = get_depth_frames(scene)
    K = np.loadtxt(os.path.join(root, scene, "camera-intrinsics.txt"))

    # Pick a random depth frame
    path = np.random.choice(frames)
    T0 = np.loadtxt(path.replace(".depth.png", ".pose.txt"))
    depth = np.array(Image.open(path)) * 0.001
    color = np.array(Image.open(path.replace(".depth.png", ".color.png")))
    depth[depth > cutoff] = 0.0
    if np.isnan(np.sum(T0)):
        return None

    # Pick a random point P
    u0 = np.random.choice(depth.shape[1])
    v0 = np.random.choice(depth.shape[0])
    if depth[v0, u0] <= 0.0:
        return None

    # Compute bounding box
    z = depth[v0, u0]
    x = (u0 - K[0, 2]) * z / K[0, 0]
    y = (v0 - K[1, 2]) * z / K[1, 1]
    p0 = np.array([x, y, z])
    q = np.matmul(T0[0:3, 0:3], p0) + T0[0:3, 3]
    b = compute_bounding_box(p0)
    b[:, 0] = np.round(b[:, 0] * K[0, 0] / b[:, 2] + K[0, 2])
    b[:, 1] = np.round(b[:, 1] * K[1, 1] / b[:, 2] + K[1, 2])

    # Get the depth patch
    x = np.array([np.min(b[:, 0]), np.max(b[:, 0])], dtype=np.int32)
    y = np.array([np.min(b[:, 1]), np.max(b[:, 1])], dtype=np.int32)
    if np.any(x < 0) or np.any(x >= depth.shape[1]):
        return None
    if np.any(y < 0) or np.any(y >= depth.shape[0]):
        return None

    patch = {}
    patch["cloud"] = extract_point_cloud(depth, color, x, y, p0, K)
    patch["color"] = extract_color_patch(color, x, y)
    patches += [patch]

    for i in range(num_samples):
        path = np.random.choice(frames)
        T1 = np.loadtxt(path.replace(".depth.png", ".pose.txt"))
        depth = np.array(Image.open(path)) * 0.001
        color = np.array(Image.open(path.replace(".depth.png", ".color.png")))
        distance = np.linalg.norm(T0[0:3, 3] - T1[0:3, 3])
        if distance < min_camera_distance:
            continue
        if np.isnan(np.sum(T1)):
            continue

        # Reproject point P into this frame
        T1 = np.linalg.inv(T1)
        p1 = np.matmul(T1[0:3, 0:3], q) + T1[0:3, 3]
        u1 = int(p1[0] * K[0, 0] / p1[2] + K[0, 2])
        v1 = int(p1[1] * K[1, 1] / p1[2] + K[1, 2])
        if u1 < 0 or u1 >= depth.shape[1]:
            continue
        if v1 < 0 or v1 >= depth.shape[0]:
            continue
        if depth[v1, u1] <= 0.0:
            continue
        if abs(depth[v1, u1] - p1[2]) > threshold:
            continue

        # Compute the second bounding box
        z = depth[v1, u1]
        x = (u1 - K[0, 2]) * z / K[0, 0]
        y = (v1 - K[1, 2]) * z / K[1, 1]
        p1 = np.array([x, y, z])
        b = compute_bounding_box(p1)
        b[:, 0] = np.round(b[:, 0] * K[0, 0] / b[:, 2] + K[0, 2])
        b[:, 1] = np.round(b[:, 1] * K[1, 1] / b[:, 2] + K[1, 2])

        # Get the matching image patch
        x = np.array([np.min(b[:, 0]), np.max(b[:, 0])], dtype=np.int32)
        y = np.array([np.min(b[:, 1]), np.max(b[:, 1])], dtype=np.int32)
        if np.any(x < 0) or np.any(x >= depth.shape[1]):
            continue
        if np.any(y < 0) or np.any(y >= depth.shape[0]):
            continue

        patch = {}
        patch["cloud"] = extract_point_cloud(depth, color, x, y, p1, K)
        patch["color"] = extract_color_patch(color, x, y)
        patches += [patch]
        break

    if len(patches) <= 1:
        return None
    return patches


def save_batch_h5(fname, batch):
    patches = list(chain(*batch))
    indices = [len(sample) for sample in batch]
    indices = np.array(indices, dtype=np.int32)
    points = np.stack([patch["cloud"] for patch in patches])
    images = np.stack([patch["color"] for patch in patches])
    fp = h5py.File(fname, "w")
    fp.create_dataset("points", data=points, compression="gzip")
    fp.create_dataset("images", data=images, compression="gzip")
    fp.close()


size = 0
batch = []
while size < dataset_size:
    scene = np.random.choice(scenes)
    sample = sample_matching_pairs(scene)
    if sample is None:
        continue

    size += 1
    batch += [sample]
    print("Sample matching patches [{}/{}]".format(size, dataset_size))

    # Save batch if needed
    if len(batch) == batch_size:
        i = size // batch_size
        fname = "data/3dmatch/h5/train/{:04d}.h5".format(i)
        print("> Saving batch to {}...".format(fname))
        save_batch_h5(fname, batch)
        batch = []
