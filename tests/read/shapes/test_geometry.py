import numpy as np
import pytest
from numpy.typing import NDArray

from dvpio.read.shapes.geometry import affine_matrix_to_shapely, apply_transformation, compute_transformation

test_cases = [
    # Scale
    (
        np.array([[0, 0], [1, 0], [0, 1]]),
        np.array([[0, 0], [2, 0], [0, 2]]),
        np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]]),
    ),
    # Translation
    (
        np.array([[0, 0], [1, 0], [0, 1]]),
        np.array([[1, 1], [2, 1], [1, 2]]),
        np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]]),
    ),
    # Rotation
    (
        np.array([[0, 0], [1, 0], [0, 1]]),
        np.array([[0, 0], [0, -1], [1, 0]]),
        np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
    ),
    # Rotate (-90degrees), scale (x2), translate (1,1)
    (
        np.array([[0, 0], [1, 0], [0, 1]]),
        np.array([[1, 1], [1, 3], [-1, 1]]),
        np.array([[0, -2, 1], [2, 0, 1], [0, 0, 1]]),
    ),
]


test_cases_shear = [
    # Add additional shear in which similarity + affine transformation differ
    (
        np.array([[0, 0], [1, 0], [0, 1]]),
        # Point 3 is sheared
        np.array([[0, 0], [1, 0], [0.5, 1]]),
        # Affine transformation
        np.array([[1.0, 0.5, -0.0], [-0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        # Similarity transformation
        np.array([[0.875, 0.25, 0.125], [-0.25, 0.875, 0.125], [0.0, 0.0, 1.0]]),
    ),
]


@pytest.mark.parametrize(["transformation_type"], [("similarity",), ("affine",)])
@pytest.mark.parametrize(["query", "reference", "affine_transformation"], test_cases)
def test_compute_transformation(
    query: NDArray[np.float64],
    reference: NDArray[np.int64],
    affine_transformation: NDArray[np.int64],
    transformation_type: str,
) -> None:
    inferred_transformation = compute_transformation(query, reference, transformation_type=transformation_type)
    assert np.isclose(inferred_transformation, affine_transformation, rtol=0.001).all()


@pytest.mark.parametrize(["query", "reference", "affine_transformation", "similarity_transformation"], test_cases_shear)
def test_compute_transformation_shear(
    query: NDArray[np.float64],
    reference: NDArray[np.int64],
    affine_transformation: NDArray[np.int64],
    similarity_transformation: NDArray[np.int64],
) -> None:
    inferred_transformation = compute_transformation(query, reference, transformation_type="affine")
    assert np.isclose(inferred_transformation, affine_transformation, rtol=0.001).all()


@pytest.mark.parametrize(["query", "reference", "affine_transformation", "similarity_transformation"], test_cases_shear)
def test_compute_similarity_transformation_shear(
    query: NDArray[np.float64],
    reference: NDArray[np.int64],
    affine_transformation: NDArray[np.int64],
    similarity_transformation: NDArray[np.int64],
) -> None:
    inferred_transformation = compute_transformation(query, reference, transformation_type="similarity")
    assert np.isclose(inferred_transformation, similarity_transformation, rtol=0.001).all()


@pytest.mark.parametrize(["query", "reference", "affine_transformation"], test_cases)
def test_apply_transformation(
    query: NDArray[np.float64],
    reference: NDArray[np.float64],
    affine_transformation: NDArray[np.float64],
) -> None:
    target = apply_transformation(query, affine_transformation)
    assert np.isclose(target, reference, rtol=0.001).all()


@pytest.mark.parametrize(
    ("affine_matrix", "expected"),
    [
        (np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]), np.array([0, 1, 3, 4, 2, 5])),
    ],
)
def test_affine_matrix_to_shapely(affine_matrix: np.ndarray, expected: np.ndarray) -> None:
    """Test that shapely convention works"""
    result = affine_matrix_to_shapely(affine_matrix=affine_matrix)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize("matrix_shape", [(1, 1), (2, 2), (3, 4), (4, 4)], ids=("1x1", "2x2", "3x4", "4x4"))
def test_affine_matrix_to_shapely__raises_incorrect_shape(matrix_shape: tuple[int, int]) -> None:
    """Test that function raises if shape is incorrect"""
    affine_matrix = np.zeros(shape=matrix_shape)
    with pytest.raises(ValueError, match="Expected matrix of shape"):
        _ = affine_matrix_to_shapely(affine_matrix=affine_matrix)
