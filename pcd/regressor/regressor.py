"""
Denoising with regression
"""

import numpy as np
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


def denoise_ls(pcd: PointCloud, threshold: float = 0.01) -> PointCloud:
    """
    Denoise the point cloud data by filtering points based on their distance from the fitted quadratic surface.

    Args:
        pcd: Input point cloud with noise
        threshold: Maximum allowed distance from the fitted surface (default: 0.01)

    Returns:
        PointCloud: Denoised point cloud with outlier points removed
    """
    coeffs = fit_quadratic(pcd)
    filtered_points = []
    for x, y, z in pcd.points:
        z_fit = (
            coeffs[0] * x**2
            + coeffs[1] * y**2
            + coeffs[2] * x * y
            + coeffs[3] * x
            + coeffs[4] * y
            + coeffs[5]
        )

        distance = abs(z - z_fit)
        if distance <= threshold:
            filtered_points.append([x, y, z])
        else:
            filtered_points.append([x, y, z_fit])
    return pointcloud(np.array(filtered_points))
