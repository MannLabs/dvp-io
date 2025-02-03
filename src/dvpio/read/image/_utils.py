from collections.abc import Callable
from typing import Any

import dask.array as da
import numpy as np
from numpy.typing import NDArray


def _create_tiles(
    dimensions: tuple[int, int],
    tile_size: tuple[int, int],
    min_coordinates: tuple[int, int] = (0, 0),
) -> NDArray[np.int_]:
    """Create rectangular tiles for a given image size.

    Parameters
    ----------
    dimensions : tuple[int, int]
        Size of the image in (width, height).
    tile_size : tuple[int, int]
        Size of individual tiles in (width, height).
    min_coordinates : tuple[int, int], optional
        Minimum coordinates in the image, defaults to (0, 0).

    Returns
    -------
    np.ndarray
        Array of shape (n_tiles_x, n_tiles_y, 4), where each entry defines a tile
        as (x, y, width, height).
    """
    # Calculate tile sizes and positions
    widths = np.full(dimensions[0] // tile_size[0], tile_size[0])
    remainder_x = dimensions[0] % tile_size[0]
    if remainder_x > 0:
        widths = np.append(widths, remainder_x)

    heights = np.full(dimensions[1] // tile_size[1], tile_size[1])
    remainder_y = dimensions[1] % tile_size[1]
    if remainder_y > 0:
        heights = np.append(heights, remainder_y)

    x_positions = min_coordinates[0] + np.cumsum(np.r_[0, widths[:-1]])
    y_positions = min_coordinates[1] + np.cumsum(np.r_[0, heights[:-1]])

    # Generate the tiles
    tiles = np.array(
        [
            [[x, y, w, h] for y, h in zip(y_positions, heights, strict=False)]
            for x, w in zip(x_positions, widths, strict=False)
        ],
        dtype=int,
    )
    return tiles


def _chunk_factory(
    func: Callable[..., NDArray], slide: Any, coords: NDArray, n_channel: int, dtype: np.dtype, **func_kwargs: Any
) -> list[list[NDArray]]:
    """Abstract factory method to tile a large microscopy image.

    Parameters
    ----------
    func
        Function to retrieve a rectangular tile from the slide image
    slide
        Slide image in format compatible with func
    coords
        Coordinates of the upper left corner of the image in formt (n_row_x, n_row_y, 4)
        where the last dimension defines the rectangular tile in format (x, y, width, height)
    n_channel
        Number of channels in array (first dimension)
    dtype
        Data type of image
    func_kwargs
        Additional keyword arguments passed to func
    """
    func_kwargs = func_kwargs if func_kwargs else {}

    # Collect each delayed chunk as item in list of list
    # Inner list becomes dim=-1 (rows)
    # Outer list becomes dim=-2 (cols)
    # see dask.array.block

    chunks = [
        [
            da.from_delayed(
                func(
                    slide,
                    coords=coords[x, y, [0, 1]],
                    size=coords[x, y, [2, 3]],
                    **func_kwargs,
                ),
                dtype=dtype,
                shape=(n_channel, *coords[x, y, [2, 3]]),
            )
            for y in range(coords.shape[1])
        ]
        for x in range(coords.shape[0])
    ]
    return chunks


def _assemble(chunks: list[list[NDArray]]) -> NDArray:
    """Assemble chunks (delayed)"""
    return da.block(chunks, allow_unknown_chunksizes=True)
