"""
Fourier-transform based denoising for point clouds
"""

import open3d as o3d
import numpy as np

from pcd.fourier.utils import (
    plane_projection,
    fourier_filter,
    grid_to_points,
    filter_fourier_artifacts,
)

if o3d.core.cuda.is_available():
    from open3d.cuda.pybind.geometry import PointCloud
else:
    from open3d.cpu.pybind.geometry import PointCloud


def denoise_fft(pc: PointCloud) -> PointCloud:
    """
    Denoise a point cloud using Fourier-transform based filtering.

    Args:
        pc: Input point cloud with noise

    Returns:
        PointCloud: Denoised point cloud

    The function projects the points to a grid, applies Fourier filtering
    to remove noise, and then reconstructs the point cloud.
    """
    points = np.asarray(pc.points)
    height, grid_x, grid_y, sampled_points = plane_projection(points)
    filtered_heights = fourier_filter(height)
    grid_points = grid_to_points(filtered_heights, grid_x, grid_y)
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grid_points))
