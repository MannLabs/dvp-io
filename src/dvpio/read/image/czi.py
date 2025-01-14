from collections.abc import Mapping
from enum import Enum
from typing import Any

import numpy as np
from dask import delayed
from numpy.typing import NDArray
from pylibCZIrw import czi as pyczi
from spatialdata.models import Image2DModel

from ._utils import _assemble, _chunk_factory, _create_tiles


class CZIPixelType(Enum):
    """Features of CZI pixel types

    Stores dimensionality, data type, and channel names of CZI pixel types
    as class for simplified access.
    Documented pixel types https://zeiss.github.io/libczi/accessors.html
    """

    Gray8 = (1, np.uint8, None)
    Gray16 = (1, np.uint16, None)
    Gray32Float = (1, np.float32, None)
    Bgr24 = (3, np.uint8, ["r", "g", "b"])
    Bgr48 = (3, np.uint16, ["r", "g", "b"])
    Bgr96Float = (3, np.float32, ["r", "g", "b"])
    Invalid = (np.nan, np.nan, np.nan)

    def __init__(self, dimensionality: int, dtype: type, c_coords: str | None) -> None:
        self.dimensionality = dimensionality
        self.dtype = dtype
        self.c_coords = c_coords

    def __lt__(self, other: "CZIPixelType") -> bool:
        """Define hierarchy of dtypes according to order of defintion"""
        if self == other:
            return False
        for elem in CZIPixelType:
            if self == elem:
                return True
            elif other == elem:
                return False
        raise ValueError("Element not in defined types")


def _parse_pixel_type(slide: pyczi.CziReader, channels: int | list[int]) -> tuple[Any, list[int]]:
    """Parse CZI channel info and return channel dimensionalities and pixel data types

    Parameters
    ----------
    slide
        CziReader, slide representation
    channels
        All channels that are supposed to be parsed


    Returns
    -------
    (CZIPixelType, list[int])
        CziPixelType: Pixeltype with the highest complexity to prevent data loss. E.g. if one channel has type uint8 and one has uint16, we parse the image to uint16
        List of dimensions: List of dimensionalities for all channels. Used to infer total dimensionality of resulting dask array

    """
    if isinstance(channels, int):
        channels = [channels]

    pixel_czi_name = [slide.get_channel_pixel_type(c) for c in channels]
    pixel_spec = [CZIPixelType[c] for c in pixel_czi_name]
    complex_pixel_spec = max(pixel_spec)

    channel_dim = [c.dimensionality for c in pixel_spec]

    return complex_pixel_spec, channel_dim


@delayed
def _get_img(
    slide: pyczi.CziReader,
    coords: tuple[int, int],
    size: tuple[int, int],
    channel: int = 0,
    timepoint: int = 0,
    z_stack: int = 0,
) -> NDArray:
    """Return numpy array of slide region

    Parameters
    ----------
    slide
        WSI
    coords
        Upper left corner (x, y) to read
    size
        Size of tile
    channel
        Channel of image
    timepoint
        Timepoint in image series (0 if only one timepoint exists)
    z_stack
        Z stack in z-series (0 if only one layer exists)

    Returns
    -------
    np.array
        Image in (c, y, x) format and RGBA channels
    """
    # czi returns an np.ndarray
    # Shape VIHT*Z*C*X*Y* (*: Obligatory) https://zeiss.github.io/libczi/imagedocumentconcept.html#autotoc_md7
    # X/Y: 2D plane
    # Z: Z-stack
    # C: Channels
    # T: Time point
    # M is used in order to enumerate all tiles in a plane i.e all planes in a given plane shall have an M-index,
    # M-index starts counting from zero to the number of tiles on that plane
    # S: Tag-like- tags images of similar interest
    img = slide.read(
        plane={"C": channel, "T": timepoint, "Z": z_stack},
        roi=(
            coords[0],  # xmin (x)
            coords[1],  # ymin (y)
            size[0],  # width (w)
            size[1],  # height (h)
        ),
    )

    # Return image in (c, y, x) format
    return np.array(img).T


def read_czi(
    path: str,
    chunk_size: tuple[int, int] = (10000, 10000),
    channels: int | list[int] = 0,
    timepoint: int = 0,
    z_stack: int = 0,
    **kwargs: Mapping[str, Any],
) -> Image2DModel:
    """Read .czi to Image2DModel

    Uses the CZI API to read .czi Carl Zeiss image format to spatialdata Image format

    Parameters
    ----------
    path
        Path to file
    chunk_size
        Size of the individual regions that are read into memory during the process
    channels
        If multiple channels are available, select these channels
    timepoint
        If timeseries, select the given index (defaults to 0 [first])
    z_stack
        If z_stack, defaults to the given stack/index (defaults to 0 [first])
    kwargs
        Keyword arguments passed to :meth:`spatialdata.models.Image2DModel.parse`

    Returns
    -------
    :class:`spatialdata.models.Image2DModel`
    """
    czidoc_r = pyczi.CziReader(path)

    # Read dimensions
    xmin, ymin, width, height = czidoc_r.total_bounding_rectangle

    # Define coordinates for chunkwise loading of the slide
    chunk_coords = _create_tiles(dimensions=(width, height), tile_size=chunk_size, min_coordinates=(xmin, ymin))

    pixel_spec, channel_dim = _parse_pixel_type(slide=czidoc_r, channels=channels)

    if isinstance(channels, list):
        # Validate that all channels are grayscale
        if not all(c == 1 for c in channel_dim):
            raise ValueError(
                f"""Not all channels in CZI file are one dimensional (dimensionalities: {channel_dim}).
                Currently, only 1D channels are supported for multi-channel images"""
            )

        chunks = [
            _chunk_factory(
                _get_img,
                slide=czidoc_r,
                coords=chunk_coords,
                n_channel=1,
                channel=c,
                timepoint=timepoint,
                z_stack=z_stack,
            )
            for c in channels
        ]
    else:
        chunks = _chunk_factory(
            _get_img,
            slide=czidoc_r,
            coords=chunk_coords,
            n_channel=sum(channel_dim),
            channel=channels,
            timepoint=timepoint,
            z_stack=z_stack,
        )

    array = _assemble(chunks)

    return Image2DModel.parse(
        array,
        dims="cyx",
        c_coords=pixel_spec.c_coords,
        **kwargs,
    )
