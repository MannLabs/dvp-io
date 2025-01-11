import geopandas as gpd
import numpy as np
import shapely
from numpy.typing import NDArray
from spatialdata.models import PointsModel, ShapesModel


def _polygon_to_array(polygon):
    return np.array(polygon.exterior.coords)


def compute_affine_transformation(
    query_points: NDArray[np.float64], reference_points: NDArray[np.float64], precision: int | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Computes the affine transformation mapping query_points to reference_points.

    .. math:
        Aq = r

    Parameters
    ----------
    query_points
        An (N, 2) array of points in the query coordinate system.
    reference_points
        An (N, 2) array of corresponding points in the reference coordinate system.
    precision
        Rounding of affine transformation matrix

    Returns
    -------
    tuple[ndarray, ndarray]
        (2, 2) array representing the rotation transformation matrix [A],
        (2, 1) array representing translation vector.
    """
    if query_points.shape != reference_points.shape:
        raise ValueError("Point sets must have the same shape.")
    if query_points.shape[1] != 2:
        raise ValueError("Points must be 2D.")
    if query_points.shape[0] < 3:
        raise ValueError("At least three points are required to compute the transformation.")

    query_points = np.concatenate([query_points, np.ones(shape=(query_points.shape[0], 1))], axis=1)
    affine_matrix, _, _, _ = np.linalg.lstsq(query_points, reference_points, rcond=None)

    if precision is not None:
        affine_matrix = np.around(affine_matrix, precision)

    rotation, translation = affine_matrix[:2, :], affine_matrix[2, :].reshape(1, -1)
    return rotation, translation


def apply_affine_transformation(
    shape: NDArray[np.float64],
    rotation: NDArray[np.float64],
    translation: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Transform shapes between coordinate systems

    Applies rotation, translation, and switch of coordinate systems to a shape,
    in this order.

    Parameters
    ----------
    shape
        (N, 2) array of points representing a polygon, with (x, y) as last dimension
    rotation
        Rotation matrix (2, 2), representing the rotation between the coordinate systems
    translation
        Translation vector (1, 2), representing a translation [=systematic shift] between the coordinate systems

    Returns
    -------
    NDArray[np.float64]
        Shape after affine transformation.
    """
    if translation is None:
        # Identity translation
        translation = np.zeros(shape=(1, 2))

    return shape @ rotation + translation


def transform_shapes(
    shapes: gpd.GeoDataFrame | ShapesModel,
    calibration_points_target: PointsModel,
    calibration_points_source: PointsModel,
    precision: int = 3,
) -> ShapesModel:
    """Apply coordinate transformation to shapes based on calibration points from a target and a source

    Computes transformation between source and target coordinates.

    Parameters
    ----------
    shapes
        Shapes in source coordinate system (usually LMD coordinates)
    calibration_points_target
        3 Calibration points in target coordinate system (usually image/pixel coordinates)
        Expects :class:`spatialdata.models.PointsModel` with calibration points in `x`/`y` column
    calibration_points_source
        3 Calibration points, matched to `calibration_points_target` in source coordinate system (usually LMD coordinates)
        Expects :class:`spatialdata.models.PointsModel` with calibration points in `x`/`y` column
    precision
        Precision of affine transformation

    Returns
    -------
    ShapesModel
        Transformed shapes in target coordinate system
    """
    PointsModel.validate(calibration_points_source)
    PointsModel.validate(calibration_points_target)

    # Convert to numpy arrays
    calibration_points_source = calibration_points_source[["x", "y"]].to_dask_array().compute()
    calibration_points_target = calibration_points_target[["x", "y"]].to_dask_array().compute()

    # Compute rotation (2x2) and translation (2x1) matrices
    rotation, translation = compute_affine_transformation(
        calibration_points_source, calibration_points_target, precision=precision
    )
    # Transform shapes
    # Iterate through shapes and apply affine transformation
    transformed_shapes = shapes["geometry"].apply(lambda shape: _polygon_to_array(shape))
    transformed_shapes = transformed_shapes.apply(
        lambda shape: apply_affine_transformation(shape, rotation=rotation, translation=translation)
    )
    transformed_shapes = transformed_shapes.apply(lambda shape: shapely.Polygon(shape))

    # Reassign as DataFrame and parse with spatialdata
    transformed_shapes = shapes.assign(geometry=transformed_shapes)

    return ShapesModel.parse(transformed_shapes)
