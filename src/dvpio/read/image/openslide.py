"""Reader for Whole Slide Images"""

import numpy as np
import openslide
from numpy.typing import NDArray
from spatialdata.models import Image2DModel

from ._utils import _assemble, _compute_chunks, _read_chunks


def _get_img(
    slide: openslide.OpenSlide,
    x0: int,
    y0: int,
    width: int,
    height: int,
    level: int,
) -> NDArray:
    """Return numpy array of slide region

    Parameters
    ----------
    slide
        WSI
    x0, y0
        Upper left corner (x, y) to read
    width, height
        Size of tile in x direction (width) and y direction (height)
    level
        Level in pyramidal image format

    Returns
    -------
    np.array
        Image in (c=4, y, x) format and RGBA channels
    """
    # Openslide returns a PILLOW image in RGBA format
    # Shape (x, y, c)
    img = slide.read_region((x0, y0), level=level, size=(width, height))

    # Pillow stores images in (y, x, c) format
    # Return image in (c=4, y, x) format
    return np.array(img).transpose(2, 0, 1)


def read_openslide(path: str, chunk_size: tuple[int, int] = (10000, 10000), pyramidal: bool = True) -> Image2DModel:
    """Read WSI to Image2DModel

    Uses openslide to read multiple pathology slide representations and parse them
    to a lazy dask array. Currently supported formats

    Tested

    - .mirax (Mirax format)

    In principle supported by openslide:

    - Aperio (.svs, .tif)
    - DICOM (.dcm)
    - Hamamatsu (.ndpi, .vms, .vmu)
    - Leica (.scn)
    - MIRAX (.mrxs)
    - Philips (.tiff)
    - Sakura (.svslide)
    - Trestle (.tif)
    - Ventana (.bif, .tif)
    - Generic tiled TIFF (.tif)

    We recommend to use the read_czi function for this format

    - Zeiss (.czi)

    Parameters
    ----------
    path
        Path to file
    chunk_size
        Size of the individual regions that are read into memory during the process in format (x, y)
    pyramidal
        Whether to create a pyramidal image with same scales as original image

    Returns
    -------
    :class:`spatialdata.models.Image2DModel`
    """
    slide = openslide.OpenSlide(path)

    # Image is represented as pyramid. Read highest resolution
    dimensions = slide.dimensions

    # Openslide represents scales in format (level[0], level[1], ...)
    # Each scale factor is represented relative to top level
    # Get downsamples in format that can be passed to Image2DModel
    scale_factors = None
    if pyramidal:
        scale_factors = [
            int(slide.level_downsamples[i] / slide.level_downsamples[i - 1])
            for i in range(1, len(slide.level_downsamples))
        ]

    # Define coordinates for chunkwise loading of the slide
    chunk_coords = _compute_chunks(dimensions=dimensions, chunk_size=chunk_size, min_coordinates=(0, 0))

    # Load chunkwise (parallelized with dask.delayed)
    chunks = _read_chunks(_get_img, slide=slide, coords=chunk_coords, n_channel=4, dtype=np.uint8, level=0)

    # Assemble into a single dask array
    array = _assemble(chunks)

    return Image2DModel.parse(
        array,
        dims="cyx",
        c_coords=["r", "g", "b", "a"],
        scale_factors=scale_factors,
        chunks=(4, *chunk_size[::-1]),
    )
