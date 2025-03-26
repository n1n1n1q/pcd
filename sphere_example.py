"""
Sphere example -- temporary file for tetsing
"""

import open3d as o3d
import numpy as np
from dpcFFT.misc.misc import sphere
from dpcFFT.data_processor.data import split, add_noise
from dpcFFT.fourier.denoise import denoise_single

if __name__ == "__main__":
    model = sphere()
    print("=== Split cloud ===")
    pc1, pc2 = split(model)
    o3d.visualization.draw_geometries([pc1])
    print("=== Noisy cloud ===")
    noisy_pc1 = add_noise(pc1, 0.1, 0.1)
    o3d.visualization.draw_geometries([noisy_pc1])
    # noisy_pc2 = add_noise(pc2, 0.1, 0.1)
    print("=== Denoised cloud ===")
    denoised_pc1 = denoise_single(noisy_pc1)
    o3d.visualization.draw_geometries([denoised_pc1])
    # denoised_pc2 = denoise_single(noisy_pc2)
