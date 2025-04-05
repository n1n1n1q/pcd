import numpy as np
import open3d as o3d
from pcd.data_processor.data import pointcloud

def fit_plane(pcd):
    """
    Fit plane to the point cloud data
    """
    X = []
    Y = []
    for x, y, z in pcd.points:
        X.append([x, y, 1])
        Y.append(z)
    X = np.array(X)
    Y = np.array(Y)
    coefficients = np.linalg.lstsq(X, Y, rcond=None)[0]
    return coefficients


def get_orthogonal_basis(v1): 
    if v1[0] != 0 or v1[1] != 0:
        v2 = np.array([-v1[1], v1[0], 0])
    else:
        v2 = np.array([0, -v1[2], v1[1]])

    v3 = np.cross(v1, v2)

    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    v3 /= np.linalg.norm(v3)

    return np.column_stack((v2, v3, v1))


def change_of_basis_denoise(pcd, denoise_function: callable):
    points = np.asarray(pcd.points)

    coeffs = fit_plane(pcd)
    a, b, _ = coeffs

    normal_vector = np.array([a, b, -1])
    new_basis = get_orthogonal_basis(normal_vector)
    transition_marix = new_basis.T

    for i, v in enumerate(points):
        points[i] = transition_marix @ v

    denoised = denoise_function(pointcloud(points))

    denoised_points = np.asarray(denoised.points)

    for i, v in enumerate(denoised_points):
        denoised_points[i] = new_basis @ v

    return pointcloud(denoised_points)


def local_denoise(pcd, n, denoise_function: callable):

    denoised_points = None
    points = np.asarray(pcd.points)

    min_ = np.min(points, axis=0)
    max_ = np.max(points, axis=0)

    x_min, y_min, _ = min_
    x_max, y_max, _ = max_

    dx = (x_max - x_min) / n
    dy = (y_max - y_min) / n

    chunks_bounds = []

    for i in range(n):
        for j in range(n):
            x_start = x_min + i * dx
            y_start = y_min + j * dy

            x_end = x_min + (i + 1) * dx
            y_end = y_min + (j + 1) * dy

            chunks_bounds.append( ((x_start, y_start), (x_end, y_end)) )

    for chunk in chunks_bounds:
        (x_start, y_start), (x_end, y_end) = chunk

        filtered_points = points[
            (points[:, 0] >= x_start) & (points[:, 0] <= x_end) &  
            (points[:, 1] >= y_start) & (points[:, 1] <= y_end)
        ]

        denoised_pcd = change_of_basis_denoise(pointcloud(filtered_points), denoise_function=denoise_function)

        if denoised_points is None:
            denoised_points = np.asarray(denoised_pcd.points)
        else:
            denoised_points = np.vstack((denoised_points, np.asarray(denoised_pcd.points)))
    
    return pointcloud(denoised_points)

