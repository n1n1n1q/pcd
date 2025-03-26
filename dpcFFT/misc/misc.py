"""
Miscalelaneous functions
"""

import open3d as o3d
import numpy as np

def sphere(n: int = 100000):
    """
    Create a point cloud of a sphere.
    """
    mesh = o3d.geometry.TriangleMesh.create_sphere()
    mesh.compute_vertex_normals()
    return mesh.sample_points_uniformly(number_of_points=n)