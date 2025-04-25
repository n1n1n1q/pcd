"""
Miscellaneous functions for point cloud generation and manipulation
"""

import open3d as o3d
import numpy as np

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
