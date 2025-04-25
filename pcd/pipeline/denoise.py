import numpy as np
import open3d as o3d
from typing import Tuple, Callable, List, Optional
from pcd.data_processor.data import pointcloud
from pcd.pipeline.utils import get_orthogonal_basis_regression, get_orthogonal_basis_pca

if o3d.core.cuda.is_available():
    from open3d.cuda.pybind.geometry import PointCloud
else:
    from open3d.cpu.pybind.geometry import PointCloud


def change_of_basis_denoise(
    pcd: PointCloud,
    denoise_function: Callable[[PointCloud], PointCloud],
    basis_function: Optional[Callable[[PointCloud], np.ndarray]],
) -> PointCloud:
    """
    Denoise a point cloud by changing basis, applying denoising, then transforming back.

    Args:
        pcd: Input point cloud
        denoise_function: Function that takes a point cloud and returns a denoised point cloud
        basis_function: Function that computes the new basis for the point cloud

    Returns:
        PointCloud: Denoised point cloud
    """
    new_basis = basis_function(pcd)
    transition_marix = new_basis.T
    points = np.asarray(pcd.points)
    for i, v in enumerate(points):
        points[i] = transition_marix @ v

    denoised = denoise_function(pointcloud(points))

    denoised_points = np.asarray(denoised.points)

    for i, v in enumerate(denoised_points):
        denoised_points[i] = new_basis @ v

    return pointcloud(denoised_points)


def local_denoise(
    pcd: PointCloud,
    n: int,
    denoise_function: Callable[[PointCloud], PointCloud],
    basis_function: str = "regression",
) -> PointCloud:
    """
    Apply denoising locally by dividing the point cloud into n×n regions.

    Args:
        pcd: Input point cloud
        n: Number of divisions along each axis (resulting in n^2 regions)
        denoise_function: Function that takes a point cloud and returns a denoised point cloud
        basis_function: Type of basis function to use, either "regression" or "pca"

    Returns:
        PointCloud: Denoised point cloud
    """
    match basis_function:
        case "regression":
            basis_function = get_orthogonal_basis_regression
        case "pca":
            basis_function = get_orthogonal_basis_pca
        case _:
            raise ValueError("Invalid basis function. Use 'regression' or 'pca'.")

    denoised_points = None
    points = np.asarray(pcd.points)

    min_ = np.min(points, axis=0)
    max_ = np.max(points, axis=0)

    x_min, y_min, _ = min_
    x_max, y_max, _ = max_

    dx = (x_max - x_min) / n
    dy = (y_max - y_min) / n

    chunks_bounds: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

    for i in range(n):
        for j in range(n):
            x_start = x_min + i * dx
            y_start = y_min + j * dy

            x_end = x_min + (i + 1) * dx
            y_end = y_min + (j + 1) * dy

            chunks_bounds.append(((x_start, y_start), (x_end, y_end)))

    for chunk in chunks_bounds:
        (x_start, y_start), (x_end, y_end) = chunk

        filtered_points = points[
            (points[:, 0] >= x_start)
            & (points[:, 0] <= x_end)
            & (points[:, 1] >= y_start)
            & (points[:, 1] <= y_end)
        ]

        if filtered_points.shape[0] > 0:
            denoised_pcd = change_of_basis_denoise(
                pointcloud(filtered_points),
                denoise_function=denoise_function,
                basis_function=basis_function,
            )

            if denoised_points is None:
                denoised_points = np.asarray(denoised_pcd.points)
            else:
                denoised_points = np.vstack(
                    (denoised_points, np.asarray(denoised_pcd.points))
                )

    return pointcloud(denoised_points)
