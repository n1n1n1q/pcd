""" """

import numpy as np
from utils import (
    compute_coordinate_system,
    plane_projection,
    fourier_filter,
    gaussian_f,
)


def denoise_multi(pc1, pc2):
    """
    Denoise and merge two point clouds.
    """
    pc1_points = np.asarray(pc1.points)
    pc2_points = np.asarray(pc2.points)

    e1, mu1 = compute_coordinate_system(pc1_points)
    e2, mu2 = compute_coordinate_system(pc2_points)

    points1_local = (pc1_points - mu1) @ e1.T
    points2_local = (pc2_points - mu2) @ e2.T

    projected1 = plane_projection(points1_local, 100)
    projected2 = plane_projection(points2_local, 100)

    filtered1 = fourier_filter(projected1, gaussian_f, 0.1)
    filtered2 = fourier_filter(projected2, gaussian_f, 0.1)

    return icp(filtered1, filtered2)


def denoise_single(pc):
    """
    Denoise a single point cloud.
    """
    points = np.asarray(pc.points)
    e, mu = compute_coordinate_system(points)
    projected = plane_projection(points, 100)

    return projected
