import numpy as np

from pcd.misc.misc import sphere
from pcd.data_processor.data import (
    visualise_pcds,
    split,
    pointcloud,
    add_noise,
    sample,
)
from pcd.fourier.denoise import denoise_single, bug_denoise_single
from pcd.eval.metrics import chamfer_distance

sphere_pcd = sphere()

# visualise_pcds(sphere_pcd)

upper_sphere, lower_sphere = split(sphere_pcd)

upper_sphere = pointcloud(np.asarray(upper_sphere.points) + np.array([0, 0, 1]))
noised_upper_sphere = add_noise(upper_sphere, 0.1, 0.1)

denoised = bug_denoise_single(noised_upper_sphere)

# visualise_pcds(noised_upper_sphere)
visualise_pcds(denoised)
