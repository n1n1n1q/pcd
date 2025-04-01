"""
Denoising script for point clouds
"""

import argparse
import pcd.data_processor.data as dpd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Point cloud denoising script")
    parser.add_argument("input", type=str, help="Input point cloud file")
    parser.add_argument("output", type=str, help="Output point cloud file")
    args = parser.parse_args()
    model = dpd.load(args.input)
    noisy_model = dpd.add_noise(model, 0.1, 0.1)
    dpd.save(args.output, noisy_model)
