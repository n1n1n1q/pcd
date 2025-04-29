import random
import numpy as np
import open3d as o3d

if o3d.core.cuda.is_available():
    from open3d.cuda.pybind.geometry import PointCloud
else:
    from open3d.cpu.pybind.geometry import PointCloud
from typing import Tuple, Callable, Optional
from pcd.pipeline.pca import pca_power_iteration
from pcd.regressor.regressor import fit_quadratic
from pcd.data_processor.data import pointcloud


def fit_plane(pcd: PointCloud) -> np.ndarray:
    """
    Fit a plane to the point cloud data.

    Args:
        pcd: Input point cloud

    Returns:
        np.ndarray: Coefficients [a, b, c] of the plane equation z = a*x + b*y + c
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


def get_orthogonal_basis_regression(pcd: PointCloud) -> np.ndarray:
    """
    Compute an orthogonal basis where the normal vector of the fitted plane is one of the basis vectors.

    Args:
        pcd: Input point cloud

    Returns:
        np.ndarray: 3x3 matrix where columns are orthogonal unit vectors forming a basis,
                    with the last column being the normalized normal vector
    """
    coeffs = fit_plane(pcd)
    a, b, _ = coeffs

    v1 = np.array([a, b, -1])

    if v1[0] != 0 or v1[1] != 0:
        v2 = np.array([-v1[1], v1[0], 0])
    else:
        v2 = np.array([0, -v1[2], v1[1]])

    v3 = np.cross(v1, v2)

    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    v3 /= np.linalg.norm(v3)

    return np.column_stack((v2, v3, v1))


def get_orthogonal_basis_pca(pcd: PointCloud) -> np.ndarray:
    """
    Compute an orthogonal basis using PCA.
    The first two vectors are the eigenvectors of the covariance matrix,
    and the third vector is the cross product of the first two.
    Args:
        pcd: Input point cloud
    Returns:
        np.ndarray: 3x3 matrix where columns are orthogonal unit vectors forming a basis
    """
    _, eigenvectors = pca_power_iteration(pcd, num_components=2)

    v1 = eigenvectors[:, 0]
    v2 = eigenvectors[:, 1]

    v3 = np.cross(v1, v2)

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v3 = v3 / np.linalg.norm(v3)

    new_basis = np.column_stack((v1, v2, v3))
    return new_basis


def change_of_basis_denoise(
    pcd: PointCloud,
    denoise_function: Callable[[PointCloud], PointCloud],
    basis_function: Optional[Callable[[PointCloud], np.ndarray]],
) -> PointCloud:
    """
    Denoise a point cloud by changing basis, applying denoising, then transforming back.

    Args:
        pcd: Input point cloud
        denoise_function: Function that takes a point cloud and returns a denoised point cloud
        basis_function: Function that computes the new basis for the point cloud

    Returns:
        PointCloud: Denoised point cloud
    """
    new_basis = basis_function(pcd)
    transition_marix = new_basis.T
    points = np.asarray(pcd.points)
    for i, v in enumerate(points):
        points[i] = transition_marix @ v

    denoised = denoise_function(pointcloud(points))

    denoised_points = np.asarray(denoised.points)

    for i, v in enumerate(denoised_points):
        denoised_points[i] = new_basis @ v

    return pointcloud(denoised_points, colors=np.asarray(pcd.colors))


def get_locality_metric(pcd):
    new_basis = get_orthogonal_basis_pca(pcd)
    transition_marix = new_basis.T
    points = np.asarray(pcd.points).copy()
    for i, v in enumerate(points):
        points[i] = transition_marix @ v

    coeffs = fit_quadratic(pcd)
    rmse = 0
    for x, y, z in pcd.points:
        z_fit = (
            coeffs[0] * x**2
            + coeffs[1] * y**2
            + coeffs[2] * x * y
            + coeffs[3] * x
            + coeffs[4] * y
            + coeffs[5]
        )
        rmse += (z - z_fit) ** 2
    return rmse / len(pcd.points)


def euclidean_segmentation(
    pcd, distance_thresh=0.1, step_size=0.2, locality_threshold=0.50
):
    """
    :param pcd: Open3D point cloud
    :param distance_thresh: Maximum distance between points in a cluster
    :param min_cluster_size: Minimum points per cluster
    :return: List of cluster point clouds
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    segments = []

    unsegmented_points = {tuple(point) for point in points}

    while unsegmented_points:
        current_point = random.choice(list(unsegmented_points))
        print(len(unsegmented_points))
        _, idx, _ = pcd_tree.search_radius_vector_3d(current_point, distance_thresh)

        try:
            score = get_locality_metric(pointcloud(points[idx]))
        except np.linalg.LinAlgError:
            unsegmented_points -= {tuple(point) for point in points[idx]}
            continue

        local_distrance_threshold = distance_thresh
        while score > locality_threshold:
            local_distrance_threshold -= step_size
            _, idx, _ = pcd_tree.search_radius_vector_3d(
                current_point, local_distrance_threshold
            )
            score = get_locality_metric(pointcloud(points[idx]))

        if len(idx) <= 2:
            unsegmented_points -= {tuple(point) for point in points[idx]}
            continue

        local_colors = colors[idx] if colors.shape[0] != 0 else None
        segments.append(
            (
                pointcloud(points[idx], colors=local_colors),
                current_point,
                local_distrance_threshold,
            )
        )
        unsegmented_points -= {tuple(point) for point in points[idx]}

    return segments


def crop_outliers(amount):

    def wrapper(local_denoised_points, centroid, radius):
        local_denoised_points = [
            point
            for point in local_denoised_points
            if np.linalg.norm(point - centroid) <= radius * (1 - amount)
        ]

        local_denoised_points = np.vstack(local_denoised_points)
        return local_denoised_points

    return wrapper
