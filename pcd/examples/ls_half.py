"""
Half sphere example using least squares regression for denoising
"""

from typing import NoReturn
import numpy as np

from pcd.misc.misc import sphere
from pcd.data_processor.data import (
    visualise_pcds,
    split,
    pointcloud,
    add_noise,
)
from pcd.regressor.regressor import denoise_ls


def main() -> NoReturn:
    """
    Main function demonstrating half sphere point cloud denoising using regression.

    Creates a sphere point cloud, splits it, keeps only the upper half,
    adds noise, applies quadratic regression denoising, and visualizes the result.
    """
    sphere_pcd = sphere()
    upper_sphere, _ = split(sphere_pcd)
    upper_sphere = pointcloud(np.asarray(upper_sphere.points) + np.array([0, 0, 1]))
    noised_upper_sphere = add_noise(upper_sphere, 0.1, 0.1)
    denoised = denoise_ls(noised_upper_sphere)
    visualise_pcds(denoised)


if __name__ == "__main__":
    main()
