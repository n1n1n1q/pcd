"""
Example script for generating and visualizing a sphere with blobs.
"""

import numpy as np
import open3d as o3d
from pcd.data_processor.data import visualise_pcds
from pcd.data_processor.data import add_noise, add_noise_inplace
from pcd.pipeline.denoise import local_denoise
from pcd.regressor.regressor import denoise_ls
from pcd.fourier.denoise import denoise_fft
from pcd.pipeline.utils import crop_outliers

np.random.seed(100)


if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud("data/Rabbit.ply")
    pcd_noised = add_noise_inplace(pcd, 1, 0.8)
    visualise_pcds(pcd_noised)
    denoised = local_denoise(
            pcd_noised,
            denoise_function=denoise_fft,
            basis_function="pca",
            distance_threshold=3,
            step_size=0.5,
            locality_threshold=1,
            post_process=crop_outliers(0.1)
        )

    pcd_noised.translate(np.array([-70, 0, 0]))
    pcd.translate(np.array([70, 0, 0]))
    visualise_pcds(pcd_noised, denoised, pcd)
    o3d.io.write_point_cloud("OUR_YOUNG_GANG.ply", denoised)


