import geopandas as gpd
import lmd.lib as pylmd
import numpy as np
import shapely
from spatialdata.models import PointsModel, ShapesModel

from .geometry import transform_shapes


def read_lmd(path: str, calibration_points_image: PointsModel, switch_orientation: bool = True) -> ShapesModel:
    """Read and parse LMD-formatted masks for the use in spatialdata

    Wrapper for pyLMD functions.

    Parameters
    ----------
    path
        Path to LMD-formatted segmentation masks in .xml format
    calibration_points_image
        Calibration points of the image as DataFrame, with 3 calibration points. Point coordinates are
        stored as seperate columns in `x` and `y` column.
    switch_orientation
        Per default, LMD is working in a (x, y) coordinate system while the image coordinates are in a (row=y, col=x)
        coordinate system. If True, transform the coordinate systems by mirroring the coordinate system at the
        main diagonal.

    Returns
    -------
    ShapesModel
        Transformed shapes in image coordinats
    """
    # Load LMD shapes with pyLMD
    lmd_shapes = pylmd.Collection()
    lmd_shapes.load(path)

    # Transform to geopandas
    shapes = gpd.GeoDataFrame(geometry=[shapely.Polygon(shape.points) for shape in lmd_shapes.shapes])

    calibration_points_lmd = gpd.GeoDataFrame(
        data={"radius": np.ones(shape=len(lmd_shapes.calibration_points))},
        geometry=[shapely.Point(point) for point in lmd_shapes.calibration_points],
    )

    calibration_points_image = gpd.GeoDataFrame(
        geometry=[shapely.Point(row["x"], row["y"]) for _, row in calibration_points_image.iterrows()]
    )

    if len(calibration_points_lmd) < 3:
        raise ValueError(f"Require at least 3 calibration points, but only received {len(calibration_points_lmd)}")
    if len(calibration_points_lmd) != len(calibration_points_image):
        raise ValueError(
            f"Number of calibration points in image ({len(calibration_points_image)})must be equal to number of calibration points in LMD file ({len(calibration_points_lmd)})"
        )

    transformed_shapes = transform_shapes(
        shapes=shapes,
        calibration_points_target=calibration_points_image,
        calibration_points_source=calibration_points_lmd,
    )

    if switch_orientation:
        # Transformation switches x/y coordinates (mirror at main diagonal)
        switch_axes = lambda geom: geom @ np.array([[0, 1], [1, 0]])
        transformed_shapes["geometry"] = transformed_shapes["geometry"].apply(
            lambda geom: shapely.transform(geom, transformation=switch_axes)
        )

    return transformed_shapes
