import numpy as np

from dpcFFT.misc.misc import sphere
from dpcFFT.data_processor.data import visualise_pcds, split, pointcloud, add_noise
from dpcFFT.fourier.denoise import denoise_single, bug_denoise_single
from dpcFFT.eval.metrics import chamfer_distance

sphere_pcd = sphere()

upper_sphere, lower_sphere = split(sphere_pcd)

upper_sphere = pointcloud(np.asarray(upper_sphere.points) + np.array([0, 0, 1]))
noised_upper_sphere = add_noise(upper_sphere, 0.1, 0.1)

denoised = bug_denoise_single(noised_upper_sphere)

visualise_pcds(noised_upper_sphere)
visualise_pcds(denoised, upper_sphere)
