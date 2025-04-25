"""
Evaluation metrics for point cloud processing
"""

import numpy as np
from typing import Union
from scipy.spatial import cKDTree
from scipy.spatial.distance import directed_hausdorff
import open3d as o3d

if o3d.core.cuda.is_available():
    from open3d.cuda.pybind.geometry import PointCloud
else:
    from open3d.cpu.pybind.geometry import PointCloud


def chamfer_distance(denoised_pcd: PointCloud, ground_truth: PointCloud) -> float:
    """
    Compute the Chamfer distance between two point clouds.

    Args:
        denoised_pcd: Denoised point cloud
        ground_truth: Ground truth point cloud

    Returns:
        float: Chamfer distance between the two point clouds

    The Chamfer distance is the sum of the average minimum distance from each point
    in the first cloud to the second cloud and vice versa.
    """
    tree1 = cKDTree(np.asarray(denoised_pcd.points))
    tree2 = cKDTree(np.asarray(ground_truth.points))

    d1, _ = tree1.query(np.asarray(ground_truth.points), k=1)
    d2, _ = tree2.query(np.asarray(denoised_pcd.points), k=1)
    return np.mean(d1) + np.mean(d2)


def hausdorff_distance(pcd1: PointCloud, pcd2: PointCloud) -> float:
    """
    Compute the Hausdorff distance between two point clouds.

    Args:
        pcd1: First point cloud
        pcd2: Second point cloud

    Returns:
        float: Hausdorff distance between the two point clouds

    The Hausdorff distance is the maximum of the directed Hausdorff distances
    from pcd1 to pcd2 and from pcd2 to pcd1.
    """
    X = np.asarray(pcd1.points)
    Y = np.asarray(pcd2.points)
    d1 = directed_hausdorff(X, Y)[0]
    d2 = directed_hausdorff(Y, X)[0]
    return max(d1, d2)
