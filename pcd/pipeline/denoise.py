import numpy as np
import open3d as o3d
from typing import Tuple, Callable, Optional
from pcd.data_processor.data import pointcloud
from pcd.pipeline.utils import (
    get_orthogonal_basis_regression,
    get_orthogonal_basis_pca,
    euclidean_segmentation,
    change_of_basis_denoise,
)
from pcd.data_processor.data import visualise_pcds


if o3d.core.cuda.is_available():
    from open3d.cuda.pybind.geometry import PointCloud
else:
    from open3d.cpu.pybind.geometry import PointCloud


def local_denoise(
    pcd: PointCloud,
    denoise_function: Callable[[PointCloud], PointCloud],
    basis_function: str = "regression",
    distance_threshold: int = 3,
    locality_threshold=0.1,
    step_size: float = 0.2,
    post_process: callable = None,
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

    segments = euclidean_segmentation(
        pcd,
        distance_thresh=distance_threshold,
        locality_threshold=locality_threshold,
        step_size=step_size,
    )

    for i, (segment, centroid, radius) in enumerate(segments):
        filtered_points = np.asarray(segment.points)
        print(f"{i}/{len(segments)}")
        if filtered_points.shape[0] > 0:
            denoised_pcd = change_of_basis_denoise(
                pointcloud(filtered_points, colors=np.asarray(segment.colors)),
                denoise_function=denoise_function,
                basis_function=basis_function,
            )                
            
            local_denoised_points = np.asarray(denoised_pcd.points)
            if post_process is not None:
                local_denoised_points = post_process(
                    local_denoised_points, centroid, radius
                )


            if denoised_points is None:
                denoised_points = local_denoised_points
                denoised_colors = np.asarray(denoised_pcd.colors)
            else:
                denoised_points = np.vstack((denoised_points, local_denoised_points))
                denoised_colors = np.vstack((denoised_colors, denoised_pcd.colors))

    return pointcloud(denoised_points, colors=denoised_colors)
