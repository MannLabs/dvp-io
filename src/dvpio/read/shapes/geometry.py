import numpy as np
from numpy.typing import NDArray
from skimage.transform import estimate_transform


def compute_affine_transformation(
    query_points: NDArray[np.float64], reference_points: NDArray[np.float64], precision: int | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Computes the affine transformation mapping query_points to reference_points.

    .. math::
        Aq = r

    Parameters
    ----------
    query_points
        An (N, 2) array of points in the query coordinate system.
    reference_points
        An (N, 2) array of corresponding points in the reference coordinate system.
    precision
        Rounding of affine transformation matrix

    Returns
    -------
    tuple[ndarray, ndarray]
        (2, 2) array representing the rotation transformation matrix [A],
        (2, 1) array representing translation vector.
    """
    if query_points.shape != reference_points.shape:
        raise ValueError("Point sets must have the same shape.")
    if query_points.shape[1] != 2:
        raise ValueError("Points must be 2D.")
    if query_points.shape[0] < 3:
        raise ValueError("At least three points are required to compute the transformation.")

    affine_matrix = estimate_transform(ttype="affine", src=query_points, dst=reference_points)

    if precision is not None:
        affine_matrix = np.around(affine_matrix, precision)

    return affine_matrix.T


def compute_similarity_transformation(
    query_points: NDArray[np.float64], reference_points: NDArray[np.float64], precision: int | None = None
):
    """Compute the similarity transformation that maps query_points to reference_points

    Compared to an affine transformation, a similarity transformation constraints the solution space
    to scaling, rotations, reflections, and translations, i.e. angles of shapes are retained.

    .. math::
            Aq = r

    Parameters
    ----------
    query_points
        An (N, 2) array of points in the query coordinate system.
    reference_points
        An (N, 2) array of corresponding points in the reference coordinate system.
    precision
        Rounding of affine transformation matrix

    Returns
    -------
    similarity_transformation
        Similarity transformation as (3 x 3) matrix

    References
    ----------
    Least-squares estimation of transformation parameters between two point patterns, Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    """
    if query_points.shape != reference_points.shape:
        raise ValueError("Point sets must have the same shape.")
    if query_points.shape[1] != 2:
        raise ValueError("Points must be 2D.")
    if query_points.shape[0] < 3:
        raise ValueError("At least three points are required to compute the transformation.")

    affine_matrix = estimate_transform(ttype="similarity", src=query_points, dst=reference_points)

    if precision is not None:
        affine_matrix = np.around(affine_matrix, precision)

    return affine_matrix.T


def apply_affine_transformation(
    shape: NDArray[np.float64],
    affine_transformation: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Transform shapes between coordinate systems

    Applies affine transformation to a shape,
    in this order.

    Parameters
    ----------
    shape
        (N, 2) array of points representing a polygon, with (x, y) as last dimension
    rotation
        Rotation matrix (2, 2), representing the rotation between the coordinate systems
    translation
        Translation vector (1, 2), representing a translation [=systematic shift] between the coordinate systems

    Returns
    -------
    NDArray[np.float64]
        Shape (N, 2) after affine transformation.
    """
    # Extend shape with ones
    shape_mod = np.hstack([shape, np.ones(shape=(shape.shape[0], 1))])
    # Apply affine transformation
    shape_transformed = shape_mod @ affine_transformation
    # Reuturn shape without padded ones
    return shape_transformed[:, :-1]
