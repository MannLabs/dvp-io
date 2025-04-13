import os
from tempfile import mkdtemp

import geopandas as gpd
import lmd.lib as pylmd
import numpy as np
import pytest
from spatialdata.models import PointsModel

from dvpio.read.shapes import read_lmd
from dvpio.write import write_lmd

calibration_points_image = PointsModel.parse(np.array([[0, 2], [2, 2], [2, 0]]))
gdf = read_lmd("./data/triangles/collection.xml", calibration_points_image=calibration_points_image)


@pytest.mark.parametrize(
    ["gdf", "calibration_points", "annotation_name_column", "annotation_well_column"],
    [
        (gdf, calibration_points_image, None, None),
        (gdf, calibration_points_image, "name", None),
        (gdf, calibration_points_image, None, "well"),
        (gdf, calibration_points_image, "name", "well"),
    ],
)
def test_write_lmd(
    gdf: gpd.GeoDataFrame,
    calibration_points: np.ndarray,
    annotation_name_column: str | None,
    annotation_well_column: str | None,
) -> None:
    path = os.path.join(mkdtemp(), "test.xml")

    write_lmd(
        path=path,
        annotation=gdf,
        calibration_points=calibration_points,
        annotation_name_column=annotation_name_column,
        annotation_well_column=annotation_well_column,
        overwrite=True,
    )


@pytest.mark.parametrize(
    ["gdf", "calibration_points", "annotation_name_column", "annotation_well_column"],
    [
        (gdf, calibration_points_image, "name", "well"),
    ],
)
def test_write_lmd_overwrite(
    gdf: gpd.GeoDataFrame,
    calibration_points: np.ndarray,
    annotation_name_column: str | None,
    annotation_well_column: str | None,
) -> None:
    """Test repeated overwriting of xml output"""
    path = os.path.join(mkdtemp(), "test.xml")

    write_lmd(
        path=path,
        annotation=gdf,
        calibration_points=calibration_points,
        annotation_name_column=annotation_name_column,
        annotation_well_column=annotation_well_column,
        overwrite=True,
    )

    # Write same file twice
    write_lmd(
        path=path,
        annotation=gdf,
        calibration_points=calibration_points,
        annotation_name_column=annotation_name_column,
        annotation_well_column=annotation_well_column,
        overwrite=True,
    )
    assert os.path.exists(path)

    # Write file without overwrite raises error
    with pytest.raises(ValueError):
        write_lmd(
            path=path,
            annotation=gdf,
            calibration_points=calibration_points,
            annotation_name_column=annotation_name_column,
            annotation_well_column=annotation_well_column,
            overwrite=False,
        )


@pytest.mark.parametrize(
    ["read_path", "calibration_points"],
    [["./data/triangles/collection.xml", calibration_points_image]],
)
def test_read_write_lmd(read_path, calibration_points):
    """Test whether dvpio-based read-write operations modify shapes in any way"""
    write_path = os.path.join(mkdtemp(), "test.xml")

    # Read in example data
    gdf = read_lmd(read_path, calibration_points_image=calibration_points, precision=3)

    # Write
    write_lmd(write_path, annotation=gdf, calibration_points=calibration_points)

    # Compare original (ref) with rewritten copy
    ref = pylmd.Collection()
    ref.load(read_path)
    ref = ref.to_geopandas()

    query = pylmd.Collection()
    query.load(write_path)
    query = query.to_geopandas()

    assert query.equals(ref)
