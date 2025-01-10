import os

import geopandas as gpd
import lmd.lib as pylmd
import numpy as np
import spatialdata as sd


def write_lmd(
    path: str,
    annotation: gpd.GeoDataFrame | sd.models.ShapesModel,
    calibration_points: sd.models.PointsModel | gpd.GeoDataFrame,
    annotation_geometry_column: str = "geometry",
    annotation_name_column: str | None = None,
    annotation_well_column: str | None = None,
    calibration_points_geometry_column: str = "geometry",
    overwrite: bool = True,
) -> None:
    """Write cell annotations to Leica-compatible .xml file

    Parameters
    ----------
    path:
        Export path for .xml
    sdata
        Spatialdata
    annotation
        Shapes (`shapely.Polygon`) to export with pyLMD
    calibration_points
        Calibration points to export with pyLMD
    annotation_geometry_column
        Column name of Shapes in `annotation` dataframe. Will be stored as coordinates of
        the Shape in the .xml file.
    annotation_name_column
        Optional. Provide column that specifies a (unique) cell name in `annotation` dataframe.
        Will be stored in as the tag of the Shape in the xml file.
    calibration_points_geometry_column
        Column name of Points in `calibration_points` dataframe
    """
    if len(calibration_points) < 3:
        raise ValueError(f"There must be at least 3 points, currently only {len(calibration_points)}")

    if os.path.exists(path) and not overwrite:
        raise ValueError(f"Path {path} exists and overwrite is False")

    # Convert calibration points dataframe to (N, 2) array for pylmd
    calibration_points = np.array(
        calibration_points[calibration_points_geometry_column].apply(lambda point: [point.x, point.y]).tolist()
    )

    # Create pylmd collection
    collection = pylmd.Collection(calibration_points=calibration_points)

    # Load annotation and optional columns
    collection.load_geopandas(
        annotation,
        geometry_column=annotation_geometry_column,
        name_column=annotation_name_column,
        well_column=annotation_well_column,
    )

    # Save
    collection.save(path)
