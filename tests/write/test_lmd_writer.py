import os
from tempfile import mkdtemp

import geopandas as gpd
import numpy as np
import pytest
import shapely
from spatialdata.models import PointsModel, ShapesModel

from dvpio.read.shapes import read_lmd
from dvpio.write import write_lmd

gdf = ShapesModel.parse(
    gpd.GeoDataFrame(
        data={"name": ["001"], "well": ["A1"]}, geometry=[shapely.Polygon([[0, 0], [0, 1], [1, 0], [0, 0]])]
    )
)

calibration_points_image = PointsModel.parse(np.array([[15, 1015], [15, 205], [1015, 15]]))


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
    assert True

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
    [
        [
            "./data/blobs/blobs/shapes/all_tiles_contours.xml",
            calibration_points_image,
        ]
    ],
)
def test_read_write_lmd(read_path, calibration_points):
    write_path = os.path.join(mkdtemp(), "test.xml")

    gdf = read_lmd(read_path, calibration_points_image=calibration_points, switch_orientation=False)

    write_lmd(write_path, annotation=gdf, calibration_points=calibration_points)

    with open(read_path) as f:
        xml_ref = f.read()

    with open(write_path) as f:
        xml_query = f.read()

    lines_ref = xml_ref.split("\n")
    lines_query = xml_query.split("\n")

    assert len(lines_ref) == len(lines_query)

    agreement = [ref == query for ref, query in zip(lines_ref, lines_query, strict=True)]
    assert all(agreement)
