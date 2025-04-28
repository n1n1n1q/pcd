"""
Submodule for generating a PLY file.
"""

from pcd.data_processor.data import visualise_pcds
from pcd.misc.misc import sphere, sphere_with_blobs, cube
from pcd.data_processor.data import add_noise, save

if __name__ == "__main__":
    sphere = sphere(n=30000)
    noised = add_noise(sphere, noise_level=0.1, noise_extra_level=0.25)
    save("sphere.ply", sphere)
    save("sphere_noised.ply", noised)

    one_blob = sphere_with_blobs(sphere, k=1, blob_radius=0.9, blob_height=0.5)
    noised_blob = add_noise(one_blob, noise_level=0.1, noise_extra_level=0.15)
    # visualise_pcds(noised_blob)
    save("sphere_blob.ply", one_blob)
    save("sphere_blob_noised.ply", noised_blob)

    three_blobs = sphere_with_blobs(sphere, k=3, blob_radius=0.9, blob_height=0.5)
    noised_three_blobs = add_noise(three_blobs, noise_level=0.1, noise_extra_level=0.15)
    # visualise_pcds(noised_three_blobs)
    save("sphere_three_blobs.ply", three_blobs)
    save("sphere_three_blobs_noised.ply", noised_three_blobs)

    many_blobs = sphere_with_blobs(sphere, k=7, blob_radius=0.5, blob_height=0.25)
    noisy_many_blobs = add_noise(many_blobs, noise_level=0.1, noise_extra_level=0.15)
    # visualise_pcds(noisy_many_blobs)
    save("sphere_hundred_blobs.ply", many_blobs)
    save("sphere_hundred_blobs_noised.ply", noisy_many_blobs)

    cube_pcd = cube(30000)
    noised_cube = add_noise(cube_pcd, noise_level=0.1, noise_extra_level=0.15)
    # visualise_pcds(noised_cube)
    save("cube.ply", cube_pcd)
    save("cube_noised.ply", noised_cube)

    cube_blob = sphere_with_blobs(cube_pcd, k=1, blob_radius=0.9, blob_height=0.5)
    noised_cube_blob = add_noise(cube_blob, noise_level=0.1, noise_extra_level=0.15)
    # visualise_pcds(noised_cube_blob)
    save("cube_blob.ply", cube_blob)
    save("cube_blob_noised.ply", noised_cube_blob)

    cube_blob_3 = sphere_with_blobs(cube_pcd, k=3, blob_radius=0.9, blob_height=0.5)
    noised_cube_blob_3 = add_noise(cube_blob_3, noise_level=0.1, noise_extra_level=0.15)
    # visualise_pcds(noised_cube_blob_3)
    save("cube_blob_3.ply", cube_blob_3)
    save("cube_blob_3_noised.ply", noised_cube_blob_3)

    cube_blob_5 = sphere_with_blobs(cube_pcd, k=5, blob_radius=0.5, blob_height=0.25)
    noised_cube_blob_5 = add_noise(cube_blob_5, noise_level=0.1, noise_extra_level=0.15)
    visualise_pcds(cube_blob_5)
    save("cube_blob_5.ply", cube_blob_5)
    save("cube_blob_5_noised.ply", noised_cube_blob_5)
