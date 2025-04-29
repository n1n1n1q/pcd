"""
Fourier utils for point cloud denoising
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy.spatial import KDTree
from scipy.spatial import ConvexHull
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from shapely.geometry import Point, Polygon
import open3d as o3d


def compute_coordinate_system(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the principal coordinate system for a set of points.

    Args:
        points: Input points as a numpy array of shape (n, 3)

    Returns:
        Tuple containing:
            - vectors: Principal axes as rows of a 3x3 matrix
            - mean: Center of the point cloud
    """
    mean = np.mean(points, axis=0)
    centered_points = points - mean
    covariance = np.cov(centered_points.T)
    vals, vectors = np.linalg.eigh(covariance)
    idx = np.argsort(vals)[::-1]
    vectors = vectors[:, idx]
    return vectors.T, mean


def plane_projection(
    points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D points onto a regular 2D grid.

    Args:
        points: Input points as a numpy array of shape (n, 3)
        size: Grid size (grid will be size x size)

    Returns:
        Tuple containing:
            - grid: Height field values as a 2D array
            - grid_x: X coordinates of the grid
            - grid_y: Y coordinates of the grid
            - sampled_points: The 2D points that were sampled
    """
    min_ = np.min(points, axis=0)
    max_ = np.max(points, axis=0)

    dx = max_[0] - min_[0]        
    size = int(dx / 0.03)

    grid_x, grid_y = np.linspace(min_[0], max_[0], size), np.linspace(
        min_[1], max_[1], size
    )

    grid = np.zeros((size, size))
    sampled_points = []

    tree = KDTree(points[:, :2])
    for i, x in enumerate(grid_x):
        for j, y in enumerate(grid_y):
            _, idx = tree.query([x, y])
            sampled_points.append(points[idx, :2])
            grid[i, j] = points[idx, 2]

    sampled_points = np.asarray(sampled_points)

    return grid, grid_x, grid_y, sampled_points


def fourier_filter(height: np.ndarray, radius: int = 3) -> np.ndarray:
    """
    Apply a low-pass filter in the frequency domain.

    Args:
        height: Height field as a 2D array
        radius: Radius of the low-pass filter

    Returns:
        np.ndarray: Filtered height field
    """
    fft_height = fft2(height)
    fft_shifted = fftshift(fft_height)

    rows, cols = fft_shifted.shape
    center_row, center_col = rows // 2, cols // 2

    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            if distance <= radius:
                mask[i, j] = 1

    fft_filtered = fft_shifted * mask

    fft_inverse_shifted = ifftshift(fft_filtered)
    filtered_heights = ifft2(fft_inverse_shifted).real

    return filtered_heights


def grid_to_points(
    grid: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray
) -> np.ndarray:
    """
    Convert a height field grid back to 3D points.

    Args:
        grid: Height field values as a 2D array
        grid_x: X coordinates of the grid
        grid_y: Y coordinates of the grid

    Returns:
        np.ndarray: 3D points with shape (n, 3)
    """
    X, Y = np.meshgrid(grid_x, grid_y, indexing="ij")
    points = np.column_stack([X.flatten(), Y.flatten(), grid.flatten()])
    return points


def filter_fourier_artifacts(
    sampled_points: np.ndarray, grid_points: np.ndarray
) -> np.ndarray:
    """
    Filter out artifacts outside the convex hull of the original points.

    Args:
        sampled_points: Original sampled points in 2D
        grid_points: Reconstructed 3D grid points that may contain artifacts

    Returns:
        np.ndarray: Filtered 3D points
    """
    hull = ConvexHull(sampled_points)
    hull_points = sampled_points[hull.vertices]
    polygon = Polygon(hull_points)

    filtered_points = [
        point for point in grid_points if polygon.contains(Point(*point[:2]))
    ]
    return np.asarray(filtered_points)
