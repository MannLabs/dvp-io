from collections.abc import Callable, Mapping
from typing import Any

from dask.array.image import imread as daimread
from spatialdata.models import Image2DModel


def read_custom(
    path: str,
    imread: Callable | None = None,
    **kwargs: Mapping[str, Any],
) -> Image2DModel:
    """Read a custom image file to Image2DModel

    This function might not be performant for large images.

    Uses the :function:`dask.array.image.imread` function to read any image file to dask.
    Support widely used file types, including `.tiff`.

    Pass a custom reader function to `imread`

    Parameters
    ----------
    path
        Path to file
    imread
        Custom image reading function. Function should expect a filename string
        return a numpy array (:class:`dask.array.image.imread`)
    **kwargs
        Keyword arguments passed to :class:`spatialdata.models.Image2DModel.parse`

    Returns
    -------
    :class:`spatialdata.models.Image2DModel`
    """
    # Per default, reads a glob string and has 4 dimensions
    # Remove 0th dimension as we currently do not support globstrings
    img = daimread(path, imread=imread).squeeze(0)

    return Image2DModel.parse(img, **kwargs)
