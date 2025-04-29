"""
Sphere example -- demonstration of point cloud splitting, noise addition and denoising
"""

import numpy as np
from pcd.data_processor.data import visualise_pcds
from pcd.misc.misc import sphere_with_blobs, sphere, save_point_cloud_screenshot
from pcd.data_processor.data import add_noise, pointcloud
from pcd.pipeline.denoise import local_denoise
from pcd.pipeline.utils import crop_outliers
from pcd.regressor.regressor import denoise_ls
from pcd.fourier.denoise import denoise_fft

def main() -> None:
    """
    Main function demonstrating sphere point cloud processing.

    Creates a sphere point cloud, splits it, adds noise to one half,
    and then applies Fourier denoising.
    """
    np.random.seed(100)
    pcd = sphere(n=100000)
    pcd1 = sphere_with_blobs(pcd, k=5, blob_radius=0.9, blob_height=0.5)
    sphere_noised = add_noise(pcd1, 0.1, 0.1)

    visualise_pcds(sphere_noised)

    denoised = local_denoise(
        sphere_noised,
        denoise_function=denoise_fft,
        basis_function="pca",
        distance_threshold=0.45,
        step_size=0.05,
        locality_threshold=0.3,
        post_process=crop_outliers(0.10)
    )

    visualise_pcds(denoised)
    save_point_cloud_screenshot(denoised, "blob_denoised.png")


if __name__ == "__main__":
    main()
