import numpy as np
import pytest
from numpy.typing import NDArray

from dvpio.read.shapes.geometry import affine_matrix_to_shapely, apply_transformation, compute_transformation

# TODO: Fix matrices so that transposal is unnecsary
test_cases = [
    # Scale
    (
        np.array([[0, 0], [1, 0], [0, 1]]),
        np.array([[0, 0], [2, 0], [0, 2]]),
        np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]]).T,
    ),
    # Translation
    (
        np.array([[0, 0], [1, 0], [0, 1]]),
        np.array([[1, 1], [2, 1], [1, 2]]),
        np.array([[1, 0, 0], [0, 1, 0], [1, 1, 1]]).T,
    ),
    # Rotation
    (
        np.array([[0, 0], [1, 0], [0, 1]]),
        np.array([[0, 0], [0, -1], [1, 0]]),
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).T,
    ),
    # Rotate (-90degrees), scale (x2), translate (1,1)
    (
        np.array([[0, 0], [1, 0], [0, 1]]),
        np.array([[1, 1], [1, 3], [-1, 1]]),
        np.array([[0, 2, 0], [-2, 0, 0], [1, 1, 1]]).T,
    ),
]


test_cases_shear = [
    # Add additional shear in which similarity + affine transformation differ
    (
        np.array([[0, 0], [1, 0], [0, 1]]),
        # Point 3 is sheared
        np.array([[0, 0], [1, 0], [0.5, 1]]),
        # Affine transformation
        np.array([[1.0, -0.0, 0.0], [0.5, 1.0, 0.0], [-0.0, 0.0, 1.0]]).T,
        # Similarity transformation
        np.array([[0.875, -0.25, 0.0], [0.25, 0.875, 0.0], [0.125, 0.125, 1.0]]).T,
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
    inferred_transformation = compute_transformation(query, reference, transformation_type="similarity", precision=3)
    assert np.isclose(inferred_transformation, similarity_transformation, rtol=0.001).all()


@pytest.mark.parametrize(["query", "reference", "affine_transformation"], test_cases)
def test_apply_transformation(
    query: NDArray[np.float64],
    reference: NDArray[np.float64],
    affine_transformation: NDArray[np.float64],
) -> None:
    target = apply_transformation(query, affine_transformation.T)  # TODO: Remove transposal
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
