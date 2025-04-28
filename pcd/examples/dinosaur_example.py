"""
Example script for generating and visualizing a sphere with blobs.
"""

import numpy as np
import open3d as o3d
from pcd.data_processor.data import visualise_pcds
from pcd.data_processor.data import add_gaussian_noise
from pcd.pipeline.denoise import local_denoise
from pcd.regressor.regressor import denoise_ls

np.random.seed(100)


if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud("data/Rabbit.ply")
    pcd_noised = add_gaussian_noise(pcd, 0.2)
    visualise_pcds(pcd_noised)
    denoised = local_denoise(
        pcd_noised,
        denoise_function=denoise_ls,
        basis_function="pca",
        distance_threshold=1,
        step_size=1,
        min_points=100,
    )
    visualise_pcds(denoised)
