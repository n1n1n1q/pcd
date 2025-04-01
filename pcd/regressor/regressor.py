"""
Denoising with regression
"""

import numpy as np
import open3d as o3d
from pcd.data_processor.data import pointcloud


def fit_quadratic(pcd):
    """
    Fit quadratic to the point cloud data
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


def denoise(pcd):
    """
    Denoise the point cloud data using regression
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
