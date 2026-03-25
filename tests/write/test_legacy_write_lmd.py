# TODO: Remove with dvpio v0.6.0
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely import Polygon
from spatialdata.models import PointsModel, ShapesModel

from dvpio.write import write_lmd as legacy_write_lmd


@pytest.fixture
def dummy_data() -> tuple[ShapesModel, PointsModel]:
    """Example data - calibration points and triangular shapes"""
    calibration_points_image = PointsModel.parse(np.array([[0, 1], [200, 200], [200, 0]]))
    gdf = ShapesModel.parse(gpd.GeoDataFrame(geometry=[Polygon([[0, 0], [0, 1], [1, 0]])]))
    affine_transformation = np.eye(3)

    return gdf, calibration_points_image, affine_transformation


def test_legacy_write_lmd__raises_warning(dummy_data, tmp_path: Path) -> None:
    """Test that legacy funciton warns when called"""
    xml_path = tmp_path / "test.xml"
    gdf, points, affine_transformation = dummy_data
    with pytest.warns(DeprecationWarning):
        legacy_write_lmd(
            xml_path, annotation=gdf, calibration_points=points, affine_transformation=affine_transformation
        )
