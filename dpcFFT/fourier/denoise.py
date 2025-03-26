""" """

import open3d as o3d
import numpy as np

if o3d.core.cuda.is_available():
    from open3d.cuda.pybind.geometry import PointCloud
else:
    from open3d.cpu.pybind.geometry import PointCloud

from dpcFFT.fourier.utils import (
    compute_coordinate_system,
    plane_projection,
    fourier_filter,
    gaussian_f,
    icp,
    grid_to_points,
    filter_fourier_artifacts,
)
from dpcFFT.data_processor.data import pointcloud


def denoise_multi(pc1, pc2):
    """
    Denoise and merge two point clouds.
    """
    pc1_points = np.asarray(pc1.points, dtype=np.float64)
    pc2_points = np.asarray(pc2.points, dtype=np.float64)

    e1, mu1 = compute_coordinate_system(pc1_points)
    e2, mu2 = compute_coordinate_system(pc2_points)

    points1_local = (pc1_points - mu1) @ e1.T
    points2_local = (pc2_points - mu2) @ e2.T

    projected1, grid_x1, grid_y1 = plane_projection(points1_local, 100)
    projected2, grid_x2, grid_y2 = plane_projection(points2_local, 100)

    filtered1 = fourier_filter(projected1, gaussian_f, 0.1)
    filtered2 = fourier_filter(projected2, gaussian_f, 0.1)

    new_pc1_points = grid_to_points(filtered1, grid_x1, grid_y1)
    new_pc2_points = grid_to_points(filtered2, grid_x2, grid_y2)

    transformed, _, _ = icp(new_pc1_points, new_pc2_points)
    return pointcloud(transformed)


def denoise_single(pc):
    """
    Denoise a single point cloud.
    """
    points = np.asarray(pc.points)
    e, mu = compute_coordinate_system(points)
    points_local = (points - mu) @ e.T
    projected, grid_x, grid_y = plane_projection(points_local, 100)
    filtered = fourier_filter(projected, gaussian_f, 0.1)
    grid_points = grid_to_points(filtered, grid_x, grid_y)
    transformed, _, _ = icp(points, grid_points)
    return pointcloud(transformed)


def bug_denoise_single(pc):
    points = np.asarray(pc.points)
    height, grid_x, grid_y, sampled_points = plane_projection(points, 100)
    filtered_heights = fourier_filter(height, gaussian_f, 0.01)
    grid_points = grid_to_points(filtered_heights, grid_x, grid_y)
    grid_points = filter_fourier_artifacts(sampled_points, grid_points)
    return pointcloud(grid_points)
