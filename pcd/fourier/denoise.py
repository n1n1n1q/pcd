""" """

import open3d as o3d
import numpy as np

if o3d.core.cuda.is_available():
    from open3d.cuda.pybind.geometry import PointCloud
else:
    from open3d.cpu.pybind.geometry import PointCloud

from pcd.fourier.utils import (
    compute_coordinate_system,
    plane_projection,
    fourier_filter,
    icp,
    grid_to_points,
    filter_fourier_artifacts,
)
from pcd.data_processor.data import pointcloud


def denoise_multi(pc1, pc2):
    pc1_points = np.asarray(pc1.points, dtype=np.float64)
    pc2_points = np.asarray(pc2.points, dtype=np.float64)

    e1, mu1 = compute_coordinate_system(pc1_points)
    e2, mu2 = compute_coordinate_system(pc2_points)

    points1_local = (pc1_points - mu1) @ e1.T
    points2_local = (pc2_points - mu2) @ e2.T

    projected1, grid_x1, grid_y1, _ = plane_projection(points1_local, 100)
    projected2, grid_x2, grid_y2, _ = plane_projection(points2_local, 100)

    filtered1 = fourier_filter(projected1)
    filtered2 = fourier_filter(projected2)

    new_pc1_points = grid_to_points(filtered1, grid_x1, grid_y1)
    new_pc2_points = grid_to_points(filtered2, grid_x2, grid_y2)

    transformed, _, _ = icp(new_pc1_points, new_pc2_points)
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(transformed))

def denoise_single(pc):
    points = np.asarray(pc.points)
    height, grid_x, grid_y, sampled_points = plane_projection(points, 100)
    filtered_heights = fourier_filter(height)
    grid_points = grid_to_points(filtered_heights, grid_x, grid_y)
    grid_points = filter_fourier_artifacts(sampled_points, grid_points)
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grid_points))