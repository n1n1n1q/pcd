"""
Sphere example -- demonstration of point cloud splitting, noise addition and denoising
"""

import open3d as o3d
from typing import NoReturn
from pcd.misc.misc import sphere
from pcd.data_processor.data import split, add_noise
from pcd.fourier.denoise import denoise_fft


def main() -> NoReturn:
    """
    Main function demonstrating sphere point cloud processing.

    Creates a sphere point cloud, splits it, adds noise to one half,
    and then applies Fourier denoising.
    """
    model = sphere()
    print("=== Split cloud ===")
    pc1, _ = split(model)
    o3d.visualization.draw_geometries([pc1])
    print("=== Noisy cloud ===")
    noisy_pc1 = add_noise(pc1, 0.1, 0.1)
    o3d.visualization.draw_geometries([noisy_pc1])
    print("=== Denoised cloud ===")
    denoised_pc1 = denoise_fft(noisy_pc1)
    o3d.visualization.draw_geometries([denoised_pc1])


if __name__ == "__main__":
    main()
