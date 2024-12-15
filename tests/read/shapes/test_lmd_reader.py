import geopandas as gpd
import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment as lsa
from scipy.spatial.distance import cdist
from spatialdata.models import PointsModel

from dvpio.read.shapes import read_lmd


def _get_centroid_xy(geometry: gpd.GeoSeries) -> NDArray[np.float64]:
    return np.array(geometry.apply(lambda geom: [geom.centroid.x, geom.centroid.y]).tolist())


calibration_points_image = PointsModel.parse(np.array([[1015, 15], [205, 15], [15, 1015]]))


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
    lmd_shapes = read_lmd(path, calibration_points, switch_orientation=True)
    lmd_centroids = _get_centroid_xy(lmd_shapes["geometry"])

    ground_truth = gpd.read_file(ground_truth_path)
    ground_truth_centroids = _get_centroid_xy(ground_truth["geometry"])

    distances = cdist(lmd_centroids, ground_truth_centroids)
    row, col = lsa(distances, maximize=False)

    assert isinstance(lmd_shapes, gpd.GeoDataFrame)
    # Centroids of matched shapes are much closer than shapes of all shapes
    assert np.median(distances[row, col]) < 0.05 * np.median(distances)
