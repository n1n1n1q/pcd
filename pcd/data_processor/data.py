"""
Data manipulation module for point cloud processing
"""

import open3d as o3d
import numpy as np
from typing import Tuple

if o3d.core.cuda.is_available():
    from open3d.cuda.pybind.geometry import PointCloud
else:
    from open3d.cpu.pybind.geometry import PointCloud


def load(file_path: str, sample: bool = False, sample_size: float = 0.05) -> PointCloud:
    """
    Load a PLY file and return an Open3D point cloud.

    Args:
        file_path: Path to the PLY file
        sample: Whether to downsample the point cloud
        sample_size: Voxel size for downsampling

    Returns:
        PointCloud: Loaded (and optionally downsampled) point cloud
    """
    model = o3d.io.read_point_cloud(file_path)
    if sample:
        model = model.voxel_down_sample(voxel_size=sample_size)
    return model


def save(file_path: str, model: PointCloud) -> None:
    """
    Save an Open3D point cloud to a PLY file.

    Args:
        file_path: Path where to save the PLY file
        model: Point cloud to save

    Raises:
        Exception: If saving the point cloud fails
    """
    try:
        o3d.io.write_point_cloud(file_path, model)
    except Exception as e:
        print(f"Error saving point cloud to {file_path}: {e}")


def add_noise(
    model: PointCloud, noise_level: float, noise_extra_level: float = 0.0
) -> PointCloud:
    """
    Add noise to the point cloud by moving existing points.

    Args:
        model: Input point cloud
        noise_level: Standard deviation of the Gaussian noise
        noise_extra_level: Unused parameter (kept for API compatibility)

    Returns:
        PointCloud: Point cloud with added noise
    """
    points = np.asarray(model.points).copy()
    colors = (
        np.asarray(model.colors).copy() if len(np.asarray(model.colors)) > 0 else None
    )

    num_points = points.shape[0]
    x_noise = np.random.normal(
        0, noise_level * 0.3, num_points
    )  # Less noise in x-direction
    y_noise = np.random.normal(
        0, noise_level * 0.3, num_points
    )  # Less noise in y-direction
    z_noise = np.random.normal(0, noise_level, num_points)  # Full noise in z-direction

    noise = np.zeros_like(points)
    noise[:, 0] = x_noise
    noise[:, 1] = y_noise
    noise[:, 2] = z_noise

    points += noise

    noisy_model = pointcloud(points, colors=colors)
    return noisy_model


def add_new_noise(
    model: PointCloud, noise_level: float, noise_extra_level: float
) -> PointCloud:
    """
    Add noise to the point cloud by adding extra points with noise in all dimensions.

    Args:
        model: Input point cloud
        noise_level: Standard deviation of the Gaussian noise
        noise_extra_level: Fraction of extra points to add (relative to original size)

    Returns:
        PointCloud: Point cloud with added noise
    """
    points = np.asarray(model.points)
    colors = (
        np.asarray(model.colors).copy() if len(np.asarray(model.colors)) > 0 else None
    )
    noise_size = int(points.shape[0] * noise_extra_level)

    x_noise = np.random.normal(0, noise_level * 0.3, noise_size)
    y_noise = np.random.normal(0, noise_level * 0.3, noise_size)
    z_noise = np.random.normal(0, noise_level, noise_size)

    noise = np.zeros((noise_size, 3))
    noise[:, 0] = x_noise
    noise[:, 1] = y_noise
    noise[:, 2] = z_noise

    n = np.random.choice(points.shape[0], size=noise_size, replace=True)
    n = points[n]
    results = np.concatenate((points, n + noise), axis=0)

    noisy_model = pointcloud(results, colors=colors)
    return noisy_model


def add_gaussian_noise(pcd: PointCloud, scale: float) -> PointCloud:
    """
    Add Gaussian noise to all points in the point cloud.

    Args:
        pcd: Input point cloud
        scale: Standard deviation of the Gaussian noise

    Returns:
        PointCloud: Point cloud with added Gaussian noise
    """
    points = np.array(pcd.points).copy()
    noised = PointCloud()
    noised.points = o3d.utility.Vector3dVector(
        points + np.random.normal(loc=0, scale=scale, size=points.shape)
    )
    noised.colors = o3d.utility.Vector3dVector(pcd.color)
    return pcd


def sample(model: PointCloud, sample_size: float) -> PointCloud:
    """
    Sample the point cloud by voxel downsampling.

    Args:
        model: Input point cloud
        sample_size: Voxel size for downsampling

    Returns:
        PointCloud: Downsampled point cloud
    """
    return model.voxel_down_sample(voxel_size=sample_size)


def split(model: PointCloud, frac: float = 0.5) -> Tuple[PointCloud, PointCloud]:
    """
    Split the point cloud into two parts based on z-coordinate.

    Args:
        model: Input point cloud
        frac: Fraction of the z-range to use as the splitting threshold

    Returns:
        Tuple containing:
            - Lower half of the point cloud
            - Upper half of the point cloud
    """
    points = np.asarray(model.points)
    min_z = np.min(points[:, 2])
    max_z = np.max(points[:, 2])
    threshold = min_z + (max_z - min_z) * frac

    lower_half_mask = points[:, 2] <= threshold
    upper_half_mask = points[:, 2] > threshold

    points1 = points[lower_half_mask]
    points2 = points[upper_half_mask]

    return pointcloud(points1), pointcloud(points2)


def pointcloud(points: np.ndarray, colors: np.ndarray = None) -> PointCloud:
    """
    Create an Open3D point cloud from a NumPy array.

    Args:
        points: NumPy array of points with shape (n, 3)

    Returns:
        PointCloud: Created Open3D point cloud
    """
    model = o3d.geometry.PointCloud()
    model.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        model.colors = o3d.utility.Vector3dVector(colors)
    return model


def visualise_pcds(*pcds: PointCloud) -> None:
    """
    Visualize multiple point clouds together.

    Args:
        *pcds: Variable number of point clouds to visualize
    """
    o3d.visualization.draw_geometries(pcds)
