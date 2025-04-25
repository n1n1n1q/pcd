import numpy as np
from typing import Tuple, Optional
import open3d as o3d

if o3d.core.cuda.is_available():
    from open3d.cuda.pybind.geometry import PointCloud
else:
    from open3d.cpu.pybind.geometry import PointCloud


def power_iteration(
    matrix: np.ndarray, num_iterations: int = 100, epsilon: float = 1e-8
) -> Tuple[float, np.ndarray]:
    """
    Implements the power iteration method to find the dominant eigenvalue
    and corresponding eigenvector of a matrix.

    Args:
        matrix: Input square matrix
        num_iterations: Maximum number of iterations
        epsilon: Convergence tolerance
        initial_vector: Initial guess vector. If None, a random vector is used.

    Returns:
        Tuple[float, np.ndarray]: The dominant eigenvalue and its corresponding eigenvector
    """
    n, _ = matrix.shape
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)
    eigenvalue = 0
    for _ in range(num_iterations):
        Av = matrix.dot(v)
        eigenvalue_new = v.dot(Av)
        v_new = Av / np.linalg.norm(Av)
        if np.abs(eigenvalue_new - eigenvalue) < epsilon:
            break
        v = v_new
        eigenvalue = eigenvalue_new
    return eigenvalue, v


def pca_power_iteration(
    pcd: PointCloud, num_components: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform PCA using power iteration to find the principal components.

    Args:
        pcd: Input point cloud
        num_components: Number of principal components to extract

    Returns:
        Tuple[np.ndarray, np.ndarray]: Eigenvalues and eigenvectors
                                      (each column of the eigenvector matrix is an eigenvector)
    """
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    cov_matrix = np.cov(centered_points, rowvar=False)
    eigenvalues = np.zeros(num_components)
    eigenvectors = np.zeros((cov_matrix.shape[0], num_components))

    tmp = cov_matrix.copy()
    for i in range(num_components):
        eigenvalue, eigenvector = power_iteration(tmp)
        eigenvalues[i] = eigenvalue
        eigenvectors[:, i] = eigenvector
        tmp = tmp - eigenvalue * np.outer(eigenvector, eigenvector)
    return eigenvalues, eigenvectors
