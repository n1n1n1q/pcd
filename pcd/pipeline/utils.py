import numpy as np
import open3d as o3d

if o3d.core.cuda.is_available():
    from open3d.cuda.pybind.geometry import PointCloud
else:
    from open3d.cpu.pybind.geometry import PointCloud
from pcd.pipeline.pca import pca_power_iteration
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


def euclidean_segmentation(
    pcd, distance_thresh=0.1, min_segment_size=100, step_size=0.2
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
    avg_distance = 0

    for point in points:
        _, _, dist = pcd_tree.search_knn_vector_3d(point, 2)
        avg_distance += np.sqrt(dist[1])

    avg_distance /= len(points)

    segments = []

    unsegmented_points = {tuple(point) for point in points}

    while unsegmented_points:
        current_point = next(iter(unsegmented_points))
        _, idx, _ = pcd_tree.search_radius_vector_3d(current_point, distance_thresh)

        local_distrance_threshold = distance_thresh

        while len(idx) < min_segment_size:
            local_distrance_threshold += step_size
            _, idx, _ = pcd_tree.search_radius_vector_3d(
                current_point, local_distrance_threshold
            )

        segments.append(pointcloud(points[idx]))
        unsegmented_points -= {tuple(point) for point in points[idx]}

    return segments
