"""
Fourier utils
"""

import numpy as np
from scipy.spatial import KDTree


def compute_coordinate_system(points):
    mean = np.mean(points, axis=0)
    points = points - mean
    covariance = np.cov(points.T)
    vals, vectors = np.linalg.eigh(covariance)
    idx = np.argsort(vals)[::-1]
    vectors = vectors[:, idx]
    return vectors.T, mean


def plane_projection(points, size):
    min_ = np.min(points, axis=0)
    max_ = np.max(points, axis=0)
    grid_x, grid_y = np.linspace(min_[0], max_[0], size), np.linspace(
        min_[1], max_[1], size
    )
    grid = np.zeros((size, size))

    tree = KDTree(points[:, :2])
    for i, x in enumerate(grid_x):
        for j, y in enumerate(grid_y):
            _, idx = tree.query([x, y])
            grid[i, j] = points[idx, 2]

    return grid, grid_x, grid_y


def fourier_filter(grid, filter_func, *args):
    fft_grid = np.fft.fft2(grid)
    gf = filter_func(grid, *args)
    fft_filtered = fft_grid * gf
    return np.fft.ifft2(fft_filtered).real


def gaussian_f(grid, sigma):
    x = np.linspace(-0.5, 0.5, grid.shape[0])
    y = np.linspace(-0.5, 0.5, grid.shape[1])
    X, Y = np.meshgrid(x, y)
    return np.exp(-(X**2 + Y**2) / (2 * sigma**2))


def grid_to_points(grid, grid_x, grid_y):
    X, Y = np.meshgrid(grid_x, grid_y, indexing="ij")
    points = np.column_stack([X.flatten(), Y.flatten(), grid.flatten()])
    return points


def icp(points, target, max_iter=50, max_error=1e-6):
    target_tree = KDTree(target)
    prev_error = float("inf")
    R_total = np.eye(3)
    t_total = np.zeros(3)
    
    for _ in range(max_iter):
        _, idx = target_tree.query(points)
        closest_points = target[idx]

        mu_src = np.mean(points, axis=0)
        mu_target = np.mean(closest_points, axis=0)

        src_centered = points - mu_src
        target_centered = closest_points - mu_target

        H = src_centered.T @ target_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        t = mu_target - R @ mu_src
        points = (R @ points.T).T + t
        R_total = R @ R_total
        t_total = R @ t_total + t

        mean_error = np.mean(np.linalg.norm(closest_points - points, axis=1))
        if abs(prev_error - mean_error) < max_error:
            break
        prev_error = mean_error
    
    return points, R_total, t_total
