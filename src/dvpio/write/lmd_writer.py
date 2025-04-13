import os

import lmd.lib as pylmd
import numpy as np
import shapely
import spatialdata as sd

from dvpio.read.shapes.geometry import apply_transformation


def write_lmd(
    path: str,
    annotation: sd.models.ShapesModel,
    calibration_points: sd.models.PointsModel,
    affine_transformation: np.ndarray | None = None,
    annotation_name_column: str | None = None,
    annotation_well_column: str | None = None,
    overwrite: bool = True,
) -> None:
    """Write cell annotations to Leica-compatible .xml file

    Parameters
    ----------
    path:
        Export path for .xml
    annotation
        Shapes to export with pyLMD
    calibration_points
        Calibration points in image coordinates
    affine_transformation
        Optional. Affine transformation to apply to the data to recover Leica coordinate system. If `None`,
        tries to recover the `to_lmd` coordinate transformation from the `annotation`
        :class:`spatialdata.models.ShapesModel` object
    annotation_name_column
        Optional. Provide column that specifies a (unique) cell name in `annotation`
        :class:`spatialdata.models.ShapesModel` object. Will be stored as the tag of
        the Shape.
    annotation_well_column
        Optional. Provide column that specifies a well in the `annotation`
        :class:`spatialdata.models.ShapesModel` object. Will be stored in as the `CapID` attribute of
        the Shape.
    overwrite
        Default `True`. Whether to overwrite existing data.

    Returns
    -------
    Saves to path

    Example
    -------
    .. code-block:: python

        from spatialdata.models import ShapesModel, PointsModel
        from tempfile import mkdtemp
        from dvpio.write import write_lmd

        annotation = ShapesModel.parse(
            gpd.GeoDataFrame(
                data={"name": ["001"], "well": ["A1"]}, geometry=[shapely.Polygon([[0, 0], [0, 1], [1, 0], [0, 0]])]
            )
        )

        calibration_points = PointsModel.parse(np.array([[0, 0], [1, 1], [0, 1]]))

        path = os.path.join(mkdtemp(), "test.xml")

        write_lmd(
            path=path,
            annotation=annotation,
            calibration_points=calibration_points,
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

    # Create pylmd collection
    collection = pylmd.Collection(orientation_transform=np.eye(2))
    collection.scale = 1

    # Transform annotation to leica coordinate system based on transformation
    if affine_transformation is None:
        affine_transformation = sd.transformations.get_transformation(
            annotation, to_coordinate_system="to_lmd"
        ).to_affine_matrix(("x", "y"), ("x", "y"))

    # Convert calibration points dataframe to (N, 2) array for pylmd
    calibration_points_transformed = apply_transformation(
        calibration_points[["x", "y"]].to_dask_array().compute(), affine_transformation
    )

    annotation_transformed = annotation["geometry"].apply(
        lambda shape: shapely.transform(
            shape,
            transformation=lambda geom: apply_transformation(geom, affine_transformation),
        )
    )

    annotation_transformed = annotation.assign(geometry=annotation_transformed)

    # Load annotation and optional columns
    collection.load_geopandas(
        annotation_transformed,
        geometry_column="geometry",
        name_column=annotation_name_column,
        well_column=annotation_well_column,
        calibration_points=calibration_points_transformed,
    )

    # Save
    collection.save(path)
