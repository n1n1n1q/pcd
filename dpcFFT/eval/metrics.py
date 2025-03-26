import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import directed_hausdorff


def chamfer_distance(denoised_pcd, ground_truth):
    tree1 = cKDTree(np.asarray(denoised_pcd.points))
    tree2 = cKDTree(np.asarray(ground_truth.points))

    d1, _ = tree1.query(np.asarray(ground_truth.points), k=1)
    d2, _ = tree2.query(np.asarray(denoised_pcd.points), k=1)
    return np.mean(d1) + np.mean(d2)


def hausdorff_distance(pcd1, pcd2):
    X = np.asarray(pcd1.points)
    Y = np.asarray(pcd2.points)
    d1 = directed_hausdorff(X, Y)[0]
    d2 = directed_hausdorff(Y, X)[0]
    return max(d1, d2)
