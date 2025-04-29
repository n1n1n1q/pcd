"""
Example script for generating and visualizing a sphere with blobs.
"""

import time
import numpy as np
from pcd.data_processor.data import visualise_pcds, load
from pcd.pipeline.denoise import local_denoise
from pcd.regressor.regressor import denoise_ls
from pcd.eval.metrics import hausdorff_distance, chamfer_distance
from pcd.fourier.denoise import denoise_fft
from pcd.pipeline.utils import crop_outliers

np.random.seed(100)

DENOISE_FUNC = None
BASIS_FUNC = "pca"


def denoise_and_eval(noised, ground_truth, func):
    """
    Denoise the point cloud and evaluate the denoised point cloud against the ground truth.
    """
    start_time = time.time()
    denoised = func(noised)
    end_time = time.time()
    visualise_pcds(denoised)

    print("Hausdorff distance:", hausdorff_distance(denoised, ground_truth))
    print("Chamfer distance:", chamfer_distance(denoised, ground_truth))
    print(f"Denoising time: {end_time - start_time:.2f} seconds")


def local_denoise_wrapped(pcd):
    """
    wrapper function for local denoising
    """
    denoised = local_denoise(
        pcd,
        denoise_function=DENOISE_FUNC,
        basis_function=BASIS_FUNC,
        distance_threshold=0.4,
        step_size=0.05,
        locality_threshold=0.05,
        post_process=POSTPROCESS_FUNCTION,
    )
    return denoised


if __name__ == "__main__":
    pcds = {
        "sphere": ("data/sphere.ply", "data/sphere_noised.ply"),
        "sphere_blob": ("data/sphere_blob.ply", "data/sphere_blob_noised.ply"),
        "sphere_blob_3": ("data/sphere_blob_3.ply", "data/sphere_blob_3_noised.ply"),
        "sphere_blob_7": ("data/sphere_blob_7.ply", "data/sphere_blob_7_noised.ply"),
        "cube": ("data/cube.ply", "data/cube_noised.ply"),
        "cube_blob": ("data/cube_blob.ply", "data/cube_blob_noised.ply"),
        "cube_blob_3": ("data/cube_blob_3.ply", "data/cube_blob_3_noised.ply"),
        "cube_blob_5": ("data/cube_blob_5.ply", "data/cube_blob_5_noised.ply"),
    }
    for name, (gt, pcd) in pcds.items():
        print(f"Processing {name} point cloud")
        gt = load(gt)
        noisy = load(pcd)
        print("Loaded point cloud with", len(gt.points), "points")
        visualise_pcds(gt)
        print("Loaded noisy point cloud with", len(noisy.points), "points")
        visualise_pcds(noisy)
        for func in ["fft", "ls"]:
            DENOISE_FUNC = denoise_fft if func == "fft" else denoise_ls
            # print(f"{func} global denoising")
            # denoise_and_eval(noisy, gt, DENOISE_FUNC)
            print(f"{func} local denoising (with PCA basis)")
            BASIS_FUNC = "pca"
            denoise_and_eval(noisy, gt, local_denoise_wrapped)
            print(f"{func} local denoising (with Regressor basis)")
            BASIS_FUNC = "regression"
            denoise_and_eval(noisy, gt, local_denoise_wrapped)
            print("Done processing", name)
