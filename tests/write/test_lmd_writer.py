import os
from tempfile import mkdtemp

import lmd.lib as pylmd
import numpy as np
import pytest
from spatialdata.models import PointsModel, ShapesModel

from dvpio.read.shapes import read_lmd
from dvpio.write import write_lmd


@pytest.fixture
def dummy_data() -> tuple[ShapesModel, PointsModel]:
    """Example data - calibration points and triangular shapes"""
    calibration_points_image = PointsModel.parse(np.array([[0, 2], [2, 2], [2, 0]]))
    gdf = read_lmd("./data/triangles/collection.xml", calibration_points_image=calibration_points_image)

    return gdf, calibration_points_image


@pytest.mark.parametrize(
    ["annotation_name_column", "annotation_well_column"],
    [
        (None, None),
        ("name", None),
        (None, "well"),
        ("name", "well"),
    ],
)
def test_write_lmd(
    tmp_path,
    dummy_data,
    annotation_name_column: str | None,
    annotation_well_column: str | None,
) -> None:
    path = tmp_path / "test.xml"
    gdf, calibration_points = dummy_data

    write_lmd(
        path=path,
        annotation=gdf,
        calibration_points=calibration_points,
        annotation_name_column=annotation_name_column,
        annotation_well_column=annotation_well_column,
        overwrite=True,
    )


@pytest.mark.parametrize(
    ["annotation_name_column", "annotation_well_column", "custom_attribute_columns"],
    [
        ("name", "well", "custom_A"),
        ("name", "well", ["custom_A", "custom_B"]),
        ("name", "well", None),
    ],
)
def test_write_custom_attributes(
    tmp_path,
    dummy_data,
    annotation_name_column: str | None,
    annotation_well_column: str | None,
    custom_attribute_columns: str | list[str] | None,
) -> None:
    path = tmp_path / "test.xml"
    gdf, calibration_points = dummy_data
    gdf = gdf.copy()

    if custom_attribute_columns is None:
        custom_attribute_columns_list = []
    else:
        if isinstance(custom_attribute_columns, str):
            custom_attribute_columns_list = [custom_attribute_columns]
        else:
            custom_attribute_columns_list = custom_attribute_columns

    for col in custom_attribute_columns_list:
        gdf[col] = "CUSTOM_VALUE"

    write_lmd(
        path=path,
        annotation=gdf,
        calibration_points=calibration_points,
        annotation_name_column=annotation_name_column,
        annotation_well_column=annotation_well_column,
        custom_attribute_columns=custom_attribute_columns,
        overwrite=True,
    )

    collection = pylmd.Collection()
    collection.load(path)

    assert all(col in gdf for col in custom_attribute_columns_list)
    assert all((gdf[col] == "CUSTOM_VALUE").all() for col in custom_attribute_columns_list)


@pytest.mark.parametrize(
    ["annotation_name_column", "annotation_well_column"],
    [
        ("name", "well"),
    ],
)
def test_write_lmd_overwrite(
    dummy_data,
    annotation_name_column: str | None,
    annotation_well_column: str | None,
) -> None:
    """Test repeated overwriting of xml output"""
    path = os.path.join(mkdtemp(), "test.xml")
    gdf, calibration_points = dummy_data

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
    ("read_path",),
    [("./data/triangles/collection.xml",)],
)
def test_read_write_lmd(tmp_path, dummy_data, read_path):
    """Test whether dvpio-based read-write operations modify shapes in any way"""
    write_path = tmp_path / "test.xml"
    _, calibration_points = dummy_data

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
