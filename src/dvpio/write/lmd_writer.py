import os

import lmd.lib as pylmd
import spatialdata as sd


def write_lmd(
    path: str,
    annotation: sd.models.ShapesModel,
    calibration_points: sd.models.PointsModel,
    annotation_name_column: str | None = None,
    annotation_well_column: str | None = None,
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

    Returns
    -------
    Saves to path

    Example
    -------
    .. code-block:: python
        from spatialdata.models import ShapesModel, PointsModel
        from tempfile import mkdtemp
        from dvpio.write import write_lmd

        gdf = ShapesModel.parse(
            gpd.GeoDataFrame(
                data={"name": ["001"], "well": ["A1"]}, geometry=[shapely.Polygon([[0, 0], [0, 1], [1, 0], [0, 0]])]
            )
        )

        calibration_points_image = PointsModel.parse(np.array([[15, 1015], [15, 205], [1015, 15]]))

        path = os.path.join(mkdtemp(), "test.xml")

        write_lmd(
            path=path,
            annotation=gdf,
            calibration_points=calibration_points,
            annotation_geometry_column="geometry",
            annotation_name_column=annotation_name_column,
            annotation_well_column=annotation_well_column,
            overwrite=True,
        )

    """
    # Validate input
    sd.models.ShapesModel.validate(annotation)
    sd.models.PointsModel.validate(calibration_points)

    if len(calibration_points) < 3:
        raise ValueError(f"There must be at least 3 points, currently only {len(calibration_points)}")

    if os.path.exists(path) and not overwrite:
        raise ValueError(f"Path {path} exists and overwrite is False")

    # Convert calibration points dataframe to (N, 2) array for pylmd
    calibration_points = calibration_points.to_dask_array().compute()

    # Create pylmd collection
    collection = pylmd.Collection(calibration_points=calibration_points)

    # Load annotation and optional columns
    collection.load_geopandas(
        annotation,
        geometry_column="geometry",
        name_column=annotation_name_column,
        well_column=annotation_well_column,
    )

    # Save
    collection.save(path)
