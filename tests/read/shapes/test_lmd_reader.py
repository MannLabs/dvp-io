import geopandas as gpd
import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment as lsa
from scipy.spatial.distance import cdist
from shapely import Polygon
from spatialdata.models import PointsModel, ShapesModel
from spatialdata.transformations import BaseTransformation

from dvpio.read.shapes import read_lmd, transform_shapes


def _get_centroid_xy(geometry: gpd.GeoSeries) -> NDArray[np.float64]:
    return np.array(geometry.apply(lambda geom: [geom.centroid.x, geom.centroid.y]).tolist())


calibration_points_image = PointsModel.parse(np.array([[15, 1015], [15, 205], [1015, 15]]))


def test_transform_shapes() -> None:
    # Create data
    calibration_points = PointsModel.parse(np.array([[0, 0], [1, 0], [0, 1]]))
    shape = Polygon([[0, 0], [1, 1], [0, 1]])
    shapes = ShapesModel.parse(gpd.GeoDataFrame(geometry=[shape] * 10))

    # Transform
    transformed_shapes = transform_shapes(
        shapes=shapes, calibration_points_source=calibration_points, calibration_points_target=calibration_points
    )

    assert isinstance(transformed_shapes, gpd.GeoDataFrame)
    assert len(transformed_shapes) == len(shapes)


@pytest.mark.parametrize(
    ["path", "calibration_points", "ground_truth_path"],
    [
        [
            "./data/blobs/blobs/shapes/all_tiles_contours.xml",
            calibration_points_image,
            "./data/blobs/blobs/ground_truth/binary-blobs.segmentation.geojson",
        ]
    ],
)
def test_read_lmd(path: str, calibration_points: NDArray[np.float64], ground_truth_path: str) -> None:
    lmd_shapes = read_lmd(path, calibration_points, switch_orientation=False, precision=3)
    lmd_centroids = _get_centroid_xy(lmd_shapes["geometry"])

    ground_truth = gpd.read_file(ground_truth_path)
    ground_truth_centroids = _get_centroid_xy(ground_truth["geometry"])

    distances = cdist(lmd_centroids, ground_truth_centroids)
    row, col = lsa(distances, maximize=False)

    assert isinstance(lmd_shapes, gpd.GeoDataFrame)
    # Centroids of matched shapes are much closer than shapes of all shapes
    # (can't be identical due to segmentation errors of cellpose)
    assert np.median(distances[row, col]) < 0.05 * np.median(distances)


@pytest.mark.parametrize(
    ["path", "calibration_points", "ground_truth_path"],
    [
        [
            "./data/blobs/blobs/shapes/all_tiles_contours.xml",
            calibration_points_image,
            "./data/blobs/blobs/ground_truth/binary-blobs.segmentation.geojson",
        ]
    ],
)
def test_read_lmd_transformation(path: str, calibration_points: NDArray[np.float64], ground_truth_path: str) -> None:
    lmd_shapes = read_lmd(path, calibration_points, switch_orientation=False, precision=3)

    assert "to_lmd" in lmd_shapes.attrs.get("transform")
    assert isinstance(lmd_shapes.attrs.get("transform").get("to_lmd"), BaseTransformation)
