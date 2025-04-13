from typing import Literal

import lmd.lib as pylmd
import numpy as np
import shapely
from spatialdata.models import PointsModel, ShapesModel
from spatialdata.transformations import Affine, set_transformation

from .geometry import apply_transformation, compute_transformation


def transform_shapes(
    shapes: ShapesModel,
    calibration_points_target: PointsModel,
    calibration_points_source: PointsModel,
    *,
    precision: int | None = None,
    transformation_type: Literal["similarity", "affine", "euclidean"] = "similarity",
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
    transformation_type
        - affine
            Full affine transformation (scaling, rotation/reflexion, translation, shearing). This operation does not preserve
            the angles within or distances the shapes
        - similarity (recommended)
            Similarity transformation. Compared to an affine transformation, a similarity transformation constraints
            the solution space to scaling, rotations, reflections, and translations, i.e. angles of shapes are retained.
            If you only want to map between image and microscopy coordinates only the subset of similarity transformations
            (scaling, rotation, reflection, translation) is required.
        - euclidean (Rigid transform)
            Only translation and rotation are allowed
    precision
        Rounding digit of affine transformation matrix. Small values (~6) might be necessary for numerical stability of shape transformations.

    Returns
    -------
    :class:`spatialdata.models.ShapesModel`
        Transformed shapes in target coordinate system

    Object has special attributes

    - `attrs.transformation`
        - `global`: (image coordinates)
        - `to_lmd`: Leica coordinate system transformation

    Raises
    ------
    AttributeError
        Checks validity of shapes and calibration points data formats
    """
    ShapesModel.validate(shapes)
    PointsModel.validate(calibration_points_source)
    PointsModel.validate(calibration_points_target)

    # Convert to numpy arrays
    calibration_points_source = calibration_points_source[["x", "y"]].to_dask_array().compute()
    calibration_points_target = calibration_points_target[["x", "y"]].to_dask_array().compute()

    # (Full affine transformation) Compute scaling, rotation+reflection, translation + shearing. In this case, angles are not preserved
    # (Similarity transformation) Constrain the affine transformation to scaling, rotation+reflection, translation. In this case, angles are preserved
    affine_transformation = compute_transformation(
        calibration_points_source,
        calibration_points_target,
        precision=precision,
        transformation_type=transformation_type,
    )

    affine_transformation_inverse = np.linalg.inv(affine_transformation)

    # Rounding might be required for numerical stability of shapely transformation
    if precision is not None:
        affine_transformation = np.around(affine_transformation, precision)
        affine_transformation_inverse = np.around(affine_transformation_inverse, precision)

    # Transform shapes
    # Iterate through shapes and apply affine transformation
    transformed_shapes = shapes["geometry"].apply(
        lambda shape: shapely.transform(
            shape, transformation=lambda geom: apply_transformation(geom, affine_transformation)
        )
    )

    # Reassign as DataFrame and parse with spatialdata
    transformed_shapes = ShapesModel.parse(shapes.assign(geometry=transformed_shapes))

    # Set inverse transformation as transformation to leica coordinate system
    set_transformation(
        transformed_shapes,
        transformation=Affine(affine_transformation_inverse.T, input_axes=("x", "y"), output_axes=("x", "y")),
        to_coordinate_system="to_lmd",
    )

    # Store original calibration points
    transformed_shapes.attrs["lmd_calibration_points"] = calibration_points_source

    return transformed_shapes


def read_lmd(
    path: str,
    calibration_points_image: PointsModel,
    transformation_type: Literal["similarity", "affine", "euclidean"] = "similarity",
    precision: int | None = 6,
    switch_orientation: bool = False,
) -> ShapesModel:
    """Read and parse LMD-formatted masks for the use in spatialdata

    Wrapper for pyLMD functions.

    Parameters
    ----------
    path
        Path to LMD-formatted segmentation masks in .xml format
    calibration_points_image
        Calibration points of the image as DataFrame, with 3 calibration points. Point coordinates are
        stored as seperate columns in `x` and `y` column.
    transformation_type
        - affine
            Full affine transformation (scaling, rotation/reflexion, translation, shearing). This operation does not preserve
            the angles within or distances the shapes
        - similarity (recommended)
            Similarity transformation. Compared to an affine transformation, a similarity transformation constraints
            the solution space to scaling, rotations, reflections, and translations, i.e. angles of shapes are retained.
            If you only want to map between image and microscopy coordinates only the subset of similarity transformations
            (scaling, rotation, reflection, translation) is required.
        - euclidean (Rigid transform)
            Only translation and rotation are allowed
    precision
        Default 6. Rounding of affine transformation matrix, which can be necessary for numerical stability of shape transformations.
        Passing `None` skips rounding.
    switch_orientation
        Per default, LMD is working in a (x, y) coordinate system while the image coordinates are in a (row=y, col=x)
        coordinate system. If True, transform the coordinate systems by mirroring the coordinate system at the
        main diagonal.

    Returns
    -------
    :class:`spatialdata.models.ShapesModel`
        Transformed shapes in image coordinates.

        Object has special attributes

        - `attrs.transformation`
            - `global` (image coordinates)
            - `to_lmd` Transformation back to leica coordinate system
    """
    PointsModel.validate(calibration_points_image)

    # Load LMD shapes with pyLMD
    lmd_shapes = pylmd.Collection()
    lmd_shapes.load(path)
    shapes = lmd_shapes.to_geopandas("name", "well")

    # Transform to spatialdata models
    shapes = ShapesModel.parse(shapes)
    calibration_points_lmd = PointsModel.parse(lmd_shapes.calibration_points)

    if len(calibration_points_lmd) < 3:
        raise ValueError(f"Require at least 3 calibration points, but only received {len(calibration_points_lmd)}")
    if len(calibration_points_lmd) != len(calibration_points_image):
        raise ValueError(
            f"Number of calibration points in image ({len(calibration_points_image)}) must be equal to number of calibration points in LMD file ({len(calibration_points_lmd)})"
        )

    transformed_shapes = transform_shapes(
        shapes=shapes,
        calibration_points_target=calibration_points_image,
        calibration_points_source=calibration_points_lmd,
        transformation_type=transformation_type,
        precision=precision,
    )

    if switch_orientation:
        # Transformation switches x/y coordinates (mirror at main diagonal)
        switch_axes = lambda geom: geom @ np.array([[0, 1], [1, 0]])
        transformed_shapes["geometry"] = transformed_shapes["geometry"].apply(
            lambda geom: shapely.transform(geom, transformation=switch_axes)
        )

    return transformed_shapes
