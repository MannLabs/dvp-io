import geopandas as gpd
import numpy as np
import pytest
from numpy.typing import NDArray
from shapely import Polygon
from spatialdata.models import PointsModel

from dvpio.read.shapes.geometry import apply_affine_transformation, compute_affine_transformation, transform_shapes

test_cases = [
    # Scale
    (
        np.array([[0, 0], [1, 0], [0, 1]]),
        np.array([[0, 0], [2, 0], [0, 2]]),
        np.array([[2, 0], [0, 2]]),
        np.array([[0, 0]]),
    ),
    # Translation
    (
        np.array([[0, 0], [1, 0], [0, 1]]),
        np.array([[1, 1], [2, 1], [1, 2]]),
        np.array([[1, 0], [0, 1]]),
        np.array([[1, 1]]),
    ),
    # Rotation
    (
        np.array([[0, 0], [1, 0], [0, 1]]),
        np.array([[0, 0], [0, -1], [1, 0]]),
        np.array([[0, -1], [1, 0]]),
        np.array([[0, 0]]),
    ),
    # Rotate (-90degrees), scale (x2), translate (1,1)
    (
        np.array([[0, 0], [1, 0], [0, 1]]),
        np.array([[1, 1], [1, 3], [-1, 1]]),
        np.array([[0, 2], [-2, 0]]),
        np.array([[1, 1]]),
    ),
]


@pytest.mark.parametrize(["query", "reference", "transformation", "translation"], test_cases)
def test_compute_affine_transformation(
    query: NDArray[np.float64],
    reference: NDArray[np.int64],
    transformation: NDArray[np.int64],
    translation: NDArray[np.int64],
) -> None:
    inferred_transformation, inferred_translation = compute_affine_transformation(query, reference, precision=3)
    assert np.isclose(inferred_transformation, transformation, rtol=0.001).all()
    assert np.isclose(inferred_translation, translation, rtol=0.001).all()


@pytest.mark.parametrize(["query", "reference", "transformation", "translation"], test_cases)
def test_apply_affine_transformation(
    query: NDArray[np.float64],
    reference: NDArray[np.int64],
    transformation: NDArray[np.int64],
    translation: NDArray[np.int64],
) -> None:
    target = apply_affine_transformation(query, transformation, translation)
    assert np.isclose(target, reference, rtol=0.001).all()


def test_transform_shapes() -> None:
    # Create data
    calibration_points = PointsModel.parse(np.array([[0, 0], [1, 0], [0, 1]]))
    shape = Polygon([[0, 0], [1, 1], [0, 1]])
    shapes = gpd.GeoDataFrame(geometry=[shape] * 10)

    # Transform
    transformed_shapes = transform_shapes(
        shapes=shapes, calibration_points_source=calibration_points, calibration_points_target=calibration_points
    )

    assert isinstance(transformed_shapes, gpd.GeoDataFrame)
    assert len(transformed_shapes) == len(shapes)
