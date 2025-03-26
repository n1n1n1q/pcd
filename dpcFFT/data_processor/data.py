"""
Data manipulation module
"""

import open3d as o3d
import numpy as np

if o3d.core.cuda.is_available():
    from open3d.cuda.pybind.geometry import PointCloud
else:
    from open3d.cpu.pybind.geometry import PointCloud


def load(file_path: str, sample: bool = False, sample_size: float = 0.05) -> PointCloud:
    """
    Load a PLY file and return an Open3D point cloud.
    """
    model = o3d.io.read_point_cloud(file_path)
    if sample:
        model = model.voxel_down_sample(voxel_size=sample_size)
    return model


def save(file_path: str, model: PointCloud) -> None:
    """
    Save an Open3D point cloud to a PLY file.
    """
    try:
        o3d.io.write_point_cloud(file_path, model)
    except Exception as e:
        print(f"Error saving point cloud to {file_path}: {e}")


def add_noise(
    model: PointCloud, noise_level: float, noise_extra_level: float
) -> PointCloud:
    """
    Noise the point cloud by adding extra points.
    """
    points = np.asarray(model.points)
    noise_size = int(points.shape[0] * noise_extra_level)
    noise = np.random.normal(0, noise_level, (noise_size, 3))
    n = np.random.choice(points.shape[0], size=noise_size, replace=True)
    n = points[n]
    results = np.concatenate((points, n + noise), axis=0)
    noisy_model = pointcloud(results)
    return noisy_model


def add_gaussian_noise(pcd: PointCloud, scale: float) -> PointCloud:
    """
    Add gaussian noise to the point cloud
    """
    points = np.array(pcd.points)
    pcd.points = o3d.utility.Vector3dVector(points + np.random.normal(loc=0, scale=0.01, size=points.shape))
    return pcd


def sample(model: PointCloud, sample_size: float) -> PointCloud:
    """
    Sample the point cloud by downsampling.
    """
    return model.voxel_down_sample(voxel_size=sample_size)


def split(model: PointCloud) -> tuple[PointCloud, PointCloud]:
    """
    Split the point cloud into two halves.
    """
    points = np.asarray(model.points)
    min_z = np.min(points[:, 2])
    max_z = np.max(points[:, 2])
    threshold = min_z + (max_z - min_z) / 2

    lower_half_mask = points[:, 2] <= threshold
    upper_half_mask = points[:, 2] > threshold

    points1 = points[lower_half_mask]
    points2 = points[upper_half_mask]

    return pointcloud(points1), pointcloud(points2)


def pointcloud(points: np.ndarray) -> PointCloud:
    """
    Create an Open3D point cloud from a NumPy array.
    """
    model = o3d.geometry.PointCloud()
    model.points = o3d.utility.Vector3dVector(points)
    return model


def visualise_pcd(pcd: PointCloud) -> None:
    """
    Visualize point cloud.
    """
    o3d.visualization.draw_geometries([pcd])


def visualise_pcds(*pcds: PointCloud) -> None:
    """
    Visualize multiple point clouds.
    """
    o3d.visualization.draw_geometries(pcds)
