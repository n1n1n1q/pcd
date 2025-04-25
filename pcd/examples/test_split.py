"""
Example demonstrating local denoising strategy with split regions
"""

import numpy as np
from typing import NoReturn
from pcd.misc.misc import sphere
from pcd.data_processor.data import (
    visualise_pcds,
    split,
    pointcloud,
    add_noise,
)
from pcd.pipeline.denoise import local_denoise
from pcd.regressor.regressor import denoise

# Number of regions for local denoising
n = 5


def main() -> NoReturn:
    """
    Main function demonstrating local denoising on a half sphere.

    Creates a sphere point cloud, splits it, keeps only the upper half,
    adds noise, applies local denoising with regression, and visualizes the result.
    The local denoising divides the point cloud into n×n regions.
    """
    sphere_pcd = sphere()
    upper_sphere, lower_sphere = split(sphere_pcd)
    upper_sphere = pointcloud(np.asarray(upper_sphere.points) + np.array([0, 0, 1]))
    noised_upper_sphere = add_noise(upper_sphere, 0.1, 0.1)
    visualise_pcds(noised_upper_sphere)
    denoised = local_denoise(noised_upper_sphere, n=5, denoise_function=denoise)
    visualise_pcds(denoised)


if __name__ == "__main__":
    main()
