"""Reader for CZI file format"""

from collections.abc import Mapping
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pylibCZIrw import czi as pyczi
from spatialdata.models import Image2DModel

from ._metadata import CZIImageMetadata
from ._utils import _assemble, _compute_chunks, _read_chunks


class CZIPixelType(Enum):
    """Features of CZI pixel types

    Stores dimensionality, data type, and channel names of CZI pixel types
    as class for simplified access.
    Documented pixel types https://zeiss.github.io/libczi/accessors.html
    """

    Gray8 = (1, np.uint16, None)
    Gray16 = (1, np.uint16, None)
    Gray32Float = (1, np.float32, None)
    Bgr24 = (3, np.uint16, ["b", "g", "r"])
    Bgr48 = (3, np.uint16, ["b", "g", "r"])
    Bgr96Float = (3, np.float32, ["b", "g", "r"])
    Invalid = (np.nan, np.nan, np.nan)

    def __init__(self, dimensionality: int, dtype: type, channel_names: list[str] | None) -> None:
        self.dimensionality = dimensionality
        self.dtype = dtype
        self.channel_names = channel_names

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


def _parse_pixel_type(slide: pyczi.CziReader, channels: list[int]) -> tuple[Any, list[int]]:
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
    pixel_czi_name = [slide.get_channel_pixel_type(c) for c in channels]
    pixel_spec = [CZIPixelType[c] for c in pixel_czi_name]
    complex_pixel_spec = max(pixel_spec)

    channel_dim = [c.dimensionality for c in pixel_spec]

    return complex_pixel_spec, channel_dim


def _get_img(
    slide: pyczi.CziReader,
    x0: int,
    y0: int,
    width: int,
    height: int,
    channel: int = 0,
    timepoint: int = 0,
    z_stack: int = 0,
) -> NDArray:
    """Return numpy array of slide region

    Parameters
    ----------
    slide
        WSI
    x0/y0
        Upper left corner (x0, y0) to read
    width/height
        Size of tile in x direction (width) and y direction (height)
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
    # pylibCZIrw returns an np.ndarray
    # Shape VIHT*Z*Y*X*C (*: Obligatory)
    # https://zeiss.github.io/libczi/imagedocumentconcept.html#autotoc_md7
    # https://zeiss.github.io/pylibczirw/#readkwargs
    # C: Channels (1 for Grayscale, 3 for BGR)
    # X/Y: 2D plane
    # Z: Z-stack
    # T: Time point
    # M is used in order to enumerate all tiles in a plane i.e all planes in a given plane shall have an M-index,
    # M-index starts counting from zero to the number of tiles on that plane
    # S: Tag-like- tags images of similar interest
    img = slide.read(
        plane={"C": channel, "T": timepoint, "Z": z_stack},
        roi=(
            x0,  # xmin (x)
            y0,  # ymin (y)
            width,  # width (w)
            height,  # height (h)
        ),
    )

    # Return image (y, x, c) -> (c, y, x) format
    return np.array(img).transpose(2, 0, 1)


def read_czi(
    path: str,
    chunk_size: tuple[int, int] = (10000, 10000),
    channels: int | list[int] | None = None,
    timepoint: int = 0,
    z_stack: int = 0,
    **kwargs: Mapping[str, Any],
) -> Image2DModel:
    """Read .czi to Image2DModel

    Uses the CZI API to read .czi Carl Zeiss image format to spatialdata image format.

    Parameters
    ----------
    path
        Path to file
    chunk_size
        Size of the individual regions that are read into memory during the process.
    channels
        Defaults to `None` which automatically selects all available channels. Passing the numeric index of a single or multiple channels
        subsets the data to the specified channels.
    timepoint
        If timeseries, select the given index (defaults to 0 [first])
    z_stack
        If z_stack, selects the the given z-plane (defaults to 0 [first])
    kwargs
        Keyword arguments passed to :meth:`spatialdata.models.Image2DModel.parse`

    Returns
    -------
    :class:`spatialdata.models.Image2DModel`
    """
    # Read slide
    czidoc_r = pyczi.CziReader(path)

    # Parse metadata
    czi_metadata = CZIImageMetadata(metadata=czidoc_r.metadata)

    # Chunked loading
    # Read dimensions
    xmin, ymin, width, height = czidoc_r.total_bounding_rectangle

    # Define coordinates for chunkwise loading of the slide
    chunk_coords = _compute_chunks(dimensions=(width, height), chunk_size=chunk_size, min_coordinates=(xmin, ymin))

    # We support the option to automatically extract channels from the metadata (None)
    # Pass a list of indices list[int] or a single index
    # Here, we assure that the channels variable stores list[int]
    if channels is None:
        channels = czi_metadata.channel_id
    if isinstance(channels, int):
        channels = [channels]

    pixel_spec, channel_dim = _parse_pixel_type(slide=czidoc_r, channels=channels)

    # For multiple indices, validate that all channels are grayscale
    # Stacking RGB images might lead to unexpected behaviour
    if (len(channels) > 1) and (not all(c == 1 for c in channel_dim)):
        raise ValueError(
            f"""Not all channels in CZI file are one dimensional (dimensionalities: {channel_dim}).
            Currently, only 1D channels are supported for multi-channel images"""
        )

    chunks = [
        _read_chunks(
            _get_img,
            slide=czidoc_r,
            coords=chunk_coords,
            n_channel=dimensionality,
            dtype=pixel_spec.dtype,
            channel=channel,
            timepoint=timepoint,
            z_stack=z_stack,
        )
        for channel, dimensionality in zip(channels, channel_dim, strict=True)
    ]

    array = _assemble(chunks)

    # Passed channel names (c_coords) should take precendence
    # If no channel names are passed, use pixel_specs.
    # This is useful for BRG images as it automatically sets the channel order correctly
    if (channel_names := kwargs.pop("c_coords", None)) is None:
        channel_names = pixel_spec.channel_names

    # For grayscale images, extract channel names from metadata
    # Only select channels that were also specified in the function call
    if channel_names is None:
        channel_names = np.array(czi_metadata.channel_names)[channels]

    return Image2DModel.parse(
        array,
        dims="cyx",
        c_coords=channel_names,
        **kwargs,
    )
