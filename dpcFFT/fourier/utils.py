"""
Fourier utils
"""
import numpy as np
from scipy.spatial import KDTree

def compute_coordinate_system(points):
    mean = np.mean(points, axis = 0)
    points = points - mean
    covariance = np.cov(points.T)
    vals, vectors = np.linalg.eigh(covariance)
    return vectors.T, mean

def plane_projection(points, size):
    min_ = np.min(points, axis = 0)
    max_ = np.max(points, axis = 0)
    grid_x, grid_y = np.linspace(min_[0], max_[0], size), np.linspace(min_[1], max_[1], size)
    grid = np.zeros((size, size))

    tree = KDTree(points[:, :2])
    for i, x in enumerate(grid_x):
        for j, y in enumerate(grid_y):
            _, idx = tree.query([x, y])
            grid[i, j] = points[idx, 2]

    return grid
