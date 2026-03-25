"""Convert a spatialdatat image to ome-tiff"""

import warnings
from collections.abc import Generator
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import spatialdata as sd
import tifffile
import xarray as xr
from ome_types import OME


def get_raster(raster: sd.models.Image2DModel | sd.models.Labels2DModel, level: str | None = None) -> xr.DataArray:
    """Get raster layer of spatialdata object as :class:`xr.DataArray`

    Parameters
    ----------
    image
        Either :class:`spatialdata.models.Image2DModel` or :class:`spatialdata.models.Label2DModel`
    level
        Level in multiscale image to use. Defaults to None which returns the highest level.

    Returns
    -------
    :class:`xr.DataArray`
        DataArray
    """
    if isinstance(raster, xr.DataArray):
        if level is not None:
            warnings.warn("Argument level is not None and will be ignored for `xr.DataArray`", stacklevel=2)
        return raster

    elif isinstance(raster, xr.DataTree):
        # Get first descendant (highest resolution) per default
        data_at_level = raster.get(key=level) if level is not None else raster.descendants[0]

        if data_at_level is None:
            raise KeyError(f"Level '{level}' not found in layer with levels {list(raster.children.keys())}")

        return (
            data_at_level.to_dataset()
            # xarray introduces a new dimension in which data variable are broadcasted against each other
            # This dimension is empty for spatialdata.Image2DModels.
            .to_array(dim="variable")
            .drop_vars("variable", errors="raise")
            .squeeze()
        )
    else:
        raise ValueError(f"Unknown raster type, {type(raster)}")


def _is_rgb(image: sd.models.Image2DModel) -> bool:
    """Check if an image is an RGB

    Checks if the channel names represent typical aliases for RGB images ("r"/"red", "g"/"green", "b"/"blue")

    References
    ----------
    - https://github.com/cgohlke/tifffile
    """
    RGB_ALIASES = ({"r", "g", "b"}, {"red", "green", "blue"})
    channel_names = {str(c).lower() for c in sd.models.get_channel_names(image)}

    return any(channel_names == rgb_alias for rgb_alias in RGB_ALIASES)


def _iter_tiles(array: np.ndarray | da.Array, tile_shape=(512, 512)) -> Generator[np.ndarray, None, None]:
    """Yield (y, x) tiles from a dask array for memory-efficient TIFF writing.

    Iterates over all leading dimensions (C, Z, T, ...) and yields tiles
    in the order expected by tifffile, computing only one tile at a time.
    """
    shape = array.shape
    height, width = shape[-2], shape[-1]
    tile_y, tile_x = tile_shape

    leading_shape = shape[:-2] if len(shape) > 2 else ()
    indices = np.ndindex(*leading_shape) if leading_shape else [()]

    for idx in indices:
        plane = array[idx] if leading_shape else array
        for y in range(0, height, tile_y):
            for x in range(0, width, tile_x):
                tile = plane[y : y + tile_y, x : x + tile_x]
                # xr.DataArray can store out-of-memory dask arrays or numpy arrays
                # ensure that numpy is returned.
                if isinstance(tile, da.Array):
                    tile = tile.compute()
                yield tile


def write_ome_tiff(
    path: Path,
    image: sd.models.Image2DModel,
    metadata: dict[str, Any] | None = None,
    tile_shape: tuple[int, int] = (1024, 1024),
    level: str | None = None,
    *,
    rgb: bool | None = None,
) -> None:
    """Export image data to ome-tiff

    Enables out-of-memory computation by writing the data iteratively to disk.

    Parameters
    ----------
    path
        Output path for the ome-tiff file.
    image
        Spatialdata Image2DModel. Only writes top-level for pyramidal images.
    metadata
        Metadata dictionary compatible with the `OME` schema.
    tile_shape
        (height, width) of each tile written to the TIFF image. Tile shape must be a multiple of 16.
    level
        Level in mulitscale image to write. If `None`, defaults to highest level. Is ignored for
        single-scale images.
    rgb
        Whether the image is RGB or grayscale. If `None`, infers the image type from the channel names.

    Returns
    -------
    Writes an `OME-TIFF` image to `path`

    Example
    -------

    .. code-block:: python

        # Write an image with default parameters
        write_ome_tiff(path, sdata["image"])

        # Write a multiscale image at a lower resolution level
        write_ome_tiff(path, sdata["multiscale_image"], level="scale2")

    """
    sd.models.Image2DModel().validate(image)

    # Autodetect RGB images if `rgb=None`
    if rgb is None:
        rgb = _is_rgb(image)
    photometric_type = "rgb" if rgb else "minisblack"

    image = get_raster(image, level=level)
    metadata = OME(**metadata).to_xml() if metadata is not None else None

    with tifffile.TiffWriter(path, bigtiff=True) as tw:
        tw.write(
            _iter_tiles(image.data, tile_shape),
            shape=image.data.shape,
            dtype=image.data.dtype,
            tile=tile_shape,
            photometric=photometric_type,
            subifds=None,
            metadata=None,
        )
