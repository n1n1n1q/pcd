"""
Data manipulation module
"""
import open3d as o3d
import numpy as np
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

def add_noise(model: PointCloud, noise_level: float, noise_extra_level: float) -> PointCloud:
    """
    Noise the point cloud by adding extra points.
    """
    points = np.asarray(model.points)
    noise_size = int(points.shape[0] * noise_extra_level)
    noise = np.random.normal(0, noise_level, (noise_size, 3))
    n = np.random.choice(points.shape[0], size=noise_size, replace=True)
    n = points[n]
    results = np.concatenate((points, n + noise), axis=0)
    noisy_model = o3d.geometry.PointCloud()
    noisy_model.points = o3d.utility.Vector3dVector(results)
    return noisy_model

def sample(model: PointCloud, sample_size: float) -> PointCloud:
    """
    Sample the point cloud by downsampling.
    """
    return model.voxel_down_sample(voxel_size=sample_size)