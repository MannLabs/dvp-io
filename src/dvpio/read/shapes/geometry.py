from typing import Literal

import numpy as np
from numpy.typing import NDArray
from skimage.transform import estimate_transform


def compute_transformation(
    query_points: NDArray[np.float64],
    reference_points: NDArray[np.float64],
    transformation_type: Literal["similarity", "affine", "euclidean"],
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
    transformation_type
        - affine
            Full affine transformation (scaling, rotation/reflexion, translation, shearing)
        - similarity
            Similarity transformation. Compared to an affine transformation, a similarity transformation constraints
            the solution space to scaling, rotations, reflections, and translations, i.e. angles of shapes are retained.
            precision
        - euclidean (Rigid transform)
            Only translation and rotation are allowed

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

    affine_matrix = estimate_transform(ttype=transformation_type, src=query_points, dst=reference_points).params

    return affine_matrix


def affine_matrix_to_shapely(affine_matrix: np.ndarray) -> np.ndarray:
    """Transform an affine matrix to the shapely syntax

    Examples
    --------

    .. code-block:: python

        array = [
            ['a', 'b', 'x0'],
            ['c', 'd', 'y0'],
            [0, 0, 1]
        ]

        affine_matrix_to_shapely(array)
        > ['a', 'b', 'c', 'd', 'x0', 'y0']
    """
    if affine_matrix.shape != (3, 3):
        raise ValueError(f"Expected matrix of shape (3, 3), got {affine_matrix.shape}")

    return affine_matrix[[0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 2, 2]]


def apply_transformation(
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
    affine_transformation
        Affine transformation applied to shapes

    Returns
    -------
    NDArray[np.float64]
        Shape (N, 2) after affine transformation.
    """
    # Extend shape with ones
    shape_mod = np.hstack([shape, np.ones(shape=(shape.shape[0], 1))])
    # Apply affine transformation
    shape_transformed = shape_mod @ affine_transformation
    # Return shape without padded ones
    return shape_transformed[:, :-1]
