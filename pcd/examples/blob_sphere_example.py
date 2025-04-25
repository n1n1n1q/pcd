"""
Example script for generating and visualizing a sphere with blobs.
"""

from pcd.data_processor.data import visualise_pcds
from pcd.misc.misc import sphere_with_blobs

if __name__ == "__main__":
    pcd1 = sphere_with_blobs(n=10000, k=5, blob_radius=0.9, blob_height=0.5)
    visualise_pcds(pcd1)
