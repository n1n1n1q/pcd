"""
Example script for generating and visualizing a sphere with blobs.
"""
import numpy as np
from pcd.data_processor.data import visualise_pcds
from pcd.misc.misc import sphere_with_blobs, sphere
from pcd.data_processor.data import add_noise
from pcd.pipeline.denoise import local_denoise
from pcd.regressor.regressor import denoise_ls

from pcd.eval.metrics import hausdorff_distance, chamfer_distance
from pcd.fourier.denoise import denoise_fft
import time
np.random.seed(100)

def denoise_and_eval(noised, ground_truth, func):
    """
    Denoise the point cloud and evaluate the denoised point cloud against the ground truth.
    """
    print("Noised point cloud:")
    visualise_pcds(noised)
    
    start_time = time.time()
    denoised = func(noised)
    end_time = time.time()
    
    print("Denoised point cloud:")
    visualise_pcds(denoised)
    print("Ground truth point cloud:")
    visualise_pcds(ground_truth)

    print("Hausdorff distance:", hausdorff_distance(denoised, ground_truth))
    print("Chamfer distance:", chamfer_distance(denoised, ground_truth))
    print(f"Denoising time: {end_time - start_time:.2f} seconds")

def denoise_with_ls(pcd):
    denoised = local_denoise(pcd, denoise_function=denoise_ls, basis_function="pca",
                             distance_threshold=0.5, step_size=0.4, min_points=200)
    return denoised

if __name__ == "__main__":
    pcd = sphere(n=100000)
    gt = pcd
    # gt = sphere_with_blobs(pcd, k=5, blob_radius=0.9, blob_height=0.5)
    noised = add_noise(pcd, noise_level=0.1, noise_extra_level=0.1)

    denoise_and_eval(noised, gt, denoise_with_ls)
    print("Denoising with LS done.")
    denoise_and_eval(noised, gt, denoise_fft)
    print("Denoising with Fourier done.")
    # save(pcd1, "blob_sphere.ply")
    # save(pcd, "sphere.ply")
    # visualise_pcds(pcd)
    # visualise_pcds(pcd1)
    # blob_noised = add_noise(pcd1, noise_level=0.1, noise_extra_level=0.1)
    # sphere_noised = add_noise(pcd, noise_level=0.1, noise_extra_level=0.1)
    # visualise_pcds(blob_noised)
    # visualise_pcds(sphere_noised)
    # save(blob_noised, "blob_sphere_noised.ply")
    # save(sphere_noised, "sphere_noised.ply")
