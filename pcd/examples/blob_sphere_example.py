"""
Example script for generating and visualizing a sphere with blobs.
"""

import numpy as np
from pcd.data_processor.data import visualise_pcds
from pcd.misc.misc import sphere_with_blobs, sphere, save_point_cloud_screenshot
from pcd.data_processor.data import add_noise
from pcd.pipeline.denoise import local_denoise
from pcd.regressor.regressor import denoise_ls

np.random.seed(100)

if __name__ == "__main__":
    pcd = sphere(n=100000)
    pcd1 = sphere_with_blobs(pcd, k=5, blob_radius=0.9, blob_height=0.5)
    blob_noised = add_noise(pcd1, noise_level=0.1, noise_extra_level=0.1)
    denoised = local_denoise(
        blob_noised,
        denoise_function=denoise_ls,
        basis_function="pca",
        distance_threshold=0.5,
        step_size=0.4,
        min_points=200,
    )
    visualise_pcds(denoised)
    save_point_cloud_screenshot(denoised, "blob_denoised.png")
