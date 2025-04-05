import numpy as np
from pcd.misc.misc import sphere
from pcd.data_processor.data import (
    visualise_pcds,
    split,
    pointcloud,
    add_noise,
    sample,
)
from pcd.pipeline.denoise import local_denoise
from pcd.regressor.regressor import denoise

sphere_pcd = sphere()

upper_sphere, lower_sphere = split(sphere_pcd)

upper_sphere = pointcloud(np.asarray(upper_sphere.points) + np.array([0, 0, 1]))
noised_upper_sphere = add_noise(upper_sphere, 0.1, 0.1)

n = 5

visualise_pcds(noised_upper_sphere)

denoised = local_denoise(noised_upper_sphere, n=5, denoise_function=denoise)

visualise_pcds(denoised)
