"""
Denoising with regression
"""

import numpy as np
import open3d as o3d
from pcd.data_processor.data import pointcloud, PointCloud


def fit_quadratic(pcd: PointCloud) -> np.ndarray:
    """
    Fit quadratic to the point cloud data.

    Args:
        pcd: Input point cloud data

    Returns:
        np.ndarray: Coefficients of the quadratic fit [a, b, c, d, e, f] where
        z = a*x² + b*y² + c*xy + d*x + e*y + f
    """
    X = []
    Y = []
    for x, y, z in pcd.points:
        X.append([x**2, y**2, x * y, x, y, 1])
        Y.append(z)
    X = np.array(X)
    Y = np.array(Y)
    coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
    return coeffs


def denoise(pcd: PointCloud) -> PointCloud:
    """
    Denoise the point cloud data using quadratic regression.

    Args:
        pcd: Input point cloud with noise

    Returns:
        PointCloud: Denoised point cloud where z-coordinates are fitted to a quadratic surface
    """
    coeffs = fit_quadratic(pcd)
    new_points = []
    for x, y, _ in pcd.points:
        z_new = (
            coeffs[0] * x**2
            + coeffs[1] * y**2
            + coeffs[2] * x * y
            + coeffs[3] * x
            + coeffs[4] * y
            + coeffs[5]
        )
        new_points.append([x, y, z_new])
    return pointcloud(np.array(new_points))
