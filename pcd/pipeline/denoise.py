import numpy as np
import open3d as o3d
from typing import Tuple, Callable, List, Optional
from pcd.data_processor.data import pointcloud,visualise_pcd
from pcd.pipeline.utils import get_orthogonal_basis_regression, get_orthogonal_basis_pca, euclidean_segmentation 

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

    return pointcloud(denoised_points, colors=np.asarray(pcd.colors))


def local_denoise(
    pcd: PointCloud,
    denoise_function: Callable[[PointCloud], PointCloud],
    basis_function: str = "regression",
    distance_threshold: int = 3,
    step_size: float = 0.2,
    min_points = 150
) -> PointCloud:
    """
    Apply denoising locally by dividing the point cloud into n×n regions.

    Args:
        pcd: Input point cloud
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
    denoised_colors = None

    segments = euclidean_segmentation(pcd, distance_threshold, min_points, step_size)

    for segment in segments:
        filtered_points = np.asarray(segment.points)

        if filtered_points.shape[0] > 0:
            denoised_pcd = change_of_basis_denoise(
                pointcloud(filtered_points, colors=np.asarray(segment.colors)),
                denoise_function=denoise_function,
                basis_function=basis_function,
            )
            if denoised_points is None:
                denoised_points = np.asarray(denoised_pcd.points)
                denoised_colors = np.asarray(denoised_pcd.colors)
            else:
                denoised_points = np.vstack(
                    (denoised_points, np.asarray(denoised_pcd.points))
                )
                denoised_colors = np.vstack(
                    (denoised_colors, np.asarray(denoised_pcd.colors))
                )

    return pointcloud(denoised_points, colors=denoised_colors)
