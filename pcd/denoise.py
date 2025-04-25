"""
Denoising script for point clouds
"""

import argparse
import time
import pcd.data_processor.data as dpd
from pcd.pipeline.denoise import local_denoise
from pcd.regressor.regressor import denoise
from pcd.fourier.denoise import denoise_single
from pcd.eval.metrics import chamfer_distance, hausdorff_distance

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Point cloud denoising script")
    parser.add_argument("input", "-i", type=str, help="Input point cloud file")
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output point cloud file"
    )
    parser.add_argument(
        "--ground_truth",
        "-g",
        type=str,
        default=None,
        help="Ground truth point cloud file",
    )
    parser.add_argument(
        "--approach",
        "-a",
        type=str,
        default="denoise_single",
        help="Denoising function to use (",
    )
    parser.add_argument(
        "--split",
        "-s",
        action="store_true",
        help="Split the point cloud before denoising",
    )
    args = parser.parse_args()

    start_time = time.time()
    print(f"Loading point cloud from {args.input}")
    model = dpd.load(args.input)
    print(f"Loaded point cloud with {len(model.points)} points")
    print(f"Time taken to load point cloud: {time.time() - start_time:.2f} seconds")

    if args.approach == "fft":
        denoise_function = denoise_single
    elif args.approach == "ls":
        denoise_function = denoise
    else:
        raise ValueError("Invalid denoising function specified")

    print(f"Using denoising {args.approach} approach")
    denoise_start_time = time.time()
    if args.split:
        denoised = local_denoise(model, n=5, denoise_function=denoise_function)
    else:
        denoised = denoise_function(model)
    print(f"Denoised point cloud with {len(denoised.points)} points")
    print(f"Time taken for denoising: {time.time() - denoise_start_time:.2f} seconds")

    if not args.output is None:
        save_start_time = time.time()
        print(f"Saving denoised point cloud to {args.output}")
        dpd.save(denoised, args.output)
        print(
            f"Time taken to save point cloud: {time.time() - save_start_time:.2f} seconds"
        )

    if not args.ground_truth is None:
        eval_start_time = time.time()
        print("Evaluating denoised point cloud against ground truth")
        ground_truth = dpd.load(args.ground_truth)
        print(f"Loaded ground truth point cloud with {len(ground_truth.points)} points")
        cd = chamfer_distance(denoised, ground_truth)
        print(f"Chamfer distance: {cd}")
        hd = hausdorff_distance(denoised, ground_truth)
        print(f"Hausdorff distance: {hd}")
        print(f"Time taken for evaluation: {time.time() - eval_start_time:.2f} seconds")

    print(f"Total time taken: {time.time() - start_time:.2f} seconds")
    print("Denoising complete")
