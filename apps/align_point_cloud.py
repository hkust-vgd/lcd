import os
import json
import open3d
import torch
import argparse
import numpy as np

from lcd.models import *


parser = argparse.ArgumentParser()
parser.add_argument("source", help="path to the source point cloud")
parser.add_argument("target", help="path to the target point cloud")
parser.add_argument("--logdir", help="path to the log directory")
parser.add_argument("--voxel_size", default=0.1, type=float)
parser.add_argument("--radius", default=0.15, type=float)
parser.add_argument("--num_points", default=1024, type=int)
args = parser.parse_args()

logdir = args.logdir
config = os.path.join(logdir, "config.json")
config = json.load(open(config))

device = config["device"]

fname = os.path.join(logdir, "model.pth")
print("> Loading model from {}".format(fname))
model = PointNetAutoencoder(
    config["embedding_size"],
    config["input_channels"],
    config["output_channels"],
    config["normalize"],
)
model.load_state_dict(torch.load(fname)["pointnet"])
model.to(device)
model.eval()


def extract_uniform_patches(pcd, voxel_size, radius, num_points):
    kdtree = open3d.geometry.KDTreeFlann(pcd)
    downsampled = pcd.voxel_down_sample(voxel_size)
    points = np.asarray(downsampled.points)
    patches = []
    for i in range(points.shape[0]):
        k, index, _ = kdtree.search_hybrid_vector_3d(points[i], radius, num_points)
        if k < num_points:
            index = np.random.choice(index, num_points, replace=True)
        xyz = np.asarray(pcd.points)[index]
        rgb = np.asarray(pcd.colors)[index]
        xyz = (xyz - points[i]) / radius  # normalize to local coordinates
        patch = np.concatenate([xyz, rgb], axis=1)
        patches.append(patch)
    patches = np.stack(patches, axis=0)
    return downsampled, patches


def compute_lcd_descriptors(patches, model, batch_size, device):
    batches = torch.tensor(patches, dtype=torch.float32)
    batches = torch.split(batches, batch_size)
    descriptors = []
    with torch.no_grad():
        for i, x in enumerate(batches):
            x = x.to(device)
            z = model.encode(x)
            z = z.cpu().numpy()
            descriptors.append(z)
    return np.concatenate(descriptors, axis=0)


source = open3d.io.read_point_cloud(args.source)
target = open3d.io.read_point_cloud(args.target)

source_points, source_patches = extract_uniform_patches(
    source, args.voxel_size, args.radius, args.num_points
)
source_descriptors = compute_lcd_descriptors(
    source_patches, model, batch_size=128, device=device
)
source_features = open3d.registration.Feature()
source_features.data = np.transpose(source_descriptors)
print("Extracted {} features from source".format(len(source_descriptors)))

target_points, target_patches = extract_uniform_patches(
    target, args.voxel_size, args.radius, args.num_points
)
target_descriptors = compute_lcd_descriptors(
    target_patches, model, batch_size=128, device=device
)
target_features = open3d.registration.Feature()
target_features.data = np.transpose(target_descriptors)
print("Extracted {} features from target".format(len(target_descriptors)))

threshold = 0.075
result = open3d.registration.registration_ransac_based_on_feature_matching(
    source_points,
    target_points,
    source_features,
    target_features,
    threshold,
    open3d.registration.TransformationEstimationPointToPoint(False),
    4,
    [open3d.registration.CorrespondenceCheckerBasedOnDistance(threshold)],
    open3d.registration.RANSACConvergenceCriteria(4000000, 500),
)

success = True
if result.transformation.trace() == 4.0:
    success = False

information = open3d.registration.get_information_matrix_from_point_clouds(
    source_points, target_points, threshold, result.transformation
)
n = min(len(source_points.points), len(target_points.points))
if (information[5, 5] / n) < 0.3:  # overlap threshold
    success = False

if not success:
    print("Cannot align two point clouds")
    exit(0)

print("Success!")
print("Visualizing alignment result...")
source.estimate_normals(open3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
target.estimate_normals(open3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
source.paint_uniform_color([1, 0.706, 0])
target.paint_uniform_color([0, 0.651, 0.929])
source.transform(result.transformation)
open3d.visualization.draw_geometries([source, target])
print(result.transformation)
