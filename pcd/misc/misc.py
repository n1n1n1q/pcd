"""
Miscellaneous functions for point cloud generation and manipulation
"""

import open3d as o3d
import numpy as np
from typing import Tuple, Optional

if o3d.core.cuda.is_available():
    from open3d.cuda.pybind.geometry import PointCloud
else:
    from open3d.cpu.pybind.geometry import PointCloud


def sphere(n: int = 100000) -> PointCloud:
    """
    Create a point cloud of a sphere.

    Args:
        n: Number of points to sample from the sphere

    Returns:
        PointCloud: Point cloud representing a sphere with n uniformly sampled points
    """
    mesh = o3d.geometry.TriangleMesh.create_sphere()
    mesh.compute_vertex_normals()
    return mesh.sample_points_uniformly(number_of_points=n)


def cube(n: int = 100000) -> PointCloud:
    """
    Create a point cloud of a cube.

    Args:
        n: Number of points to sample from the cube

    Returns:
        PointCloud: Point cloud representing a cube with n uniformly sampled points
    """
    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.compute_vertex_normals()
    return mesh.sample_points_uniformly(number_of_points=n)


def sphere_with_blobs(
    pcd, k: int = 5, blob_radius: float = 0.2, blob_height: float = 0.1
) -> PointCloud:
    """
    Create a point cloud with k blobs on it.

    Args:
        pcd: Input point cloud
        k: Number of blobs to add
        blob_radius: Radius of the blobs
        blob_height: Height of the blobs
    Returns:
        PointCloud: Point cloud with k blobs added
    """
    points = np.asarray(pcd.points)

    blob_centers = []
    for _ in range(k):
        vec = np.random.randn(3)
        vec = vec / np.linalg.norm(vec)
        blob_centers.append(vec)

    for center in blob_centers:
        dot_products = np.dot(points, center)
        angular_distances = np.arccos(np.clip(dot_products, -1.0, 1.0))
        blob_mask = angular_distances < blob_radius
        displacement_factors = np.exp(
            -(angular_distances[blob_mask] ** 2) / (2 * (blob_radius / 3) ** 2)
        )
        points[blob_mask] += np.outer(displacement_factors * blob_height, center)

    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd
