from collections.abc import Callable
from typing import Any

import dask.array as da
import numpy as np
from dask import delayed
from numpy.typing import NDArray


def _compute_chunk_sizes_positions(size: int, chunk: int, min_coord: int) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Calculate chunk sizes and positions for a given dimension and chunk size"""
    # All chunks have the same size except for the last one
    positions = np.arange(min_coord, min_coord + size, chunk)
    lengths = np.full_like(positions, chunk, dtype=int)

    # Last position coordinate exceeds maximum coordinate of slide
    last_position_coordinate = positions[-1] + chunk
    slide_max_coordinate = size + min_coord
    if last_position_coordinate > slide_max_coordinate:
        lengths[-1] = size + min_coord - positions[-1]

    return positions, lengths


def _compute_chunks(
    dimensions: tuple[int, int],
    chunk_size: tuple[int, int],
    min_coordinates: tuple[int, int] = (0, 0),
) -> NDArray[np.int_]:
    """Create all chunk specs for a given image and chunk size.

    Creates specifications (x, y, width, height) with (x, y) being the upper left corner
    of chunks of size chunk_size. Chunks at the edges correspond to the remainder of
    chunk size and dimensions

    Parameters
    ----------
    dimensions : tuple[int, int]
        Size of the image in (width, height).
    chunk_size : tuple[int, int]
        Size of individual tiles in (width, height).
    min_coordinates : tuple[int, int], optional
        Minimum coordinates (x, y) in the image, defaults to (0, 0).

    Returns
    -------
    np.ndarray
        Array of shape (n_tiles_x, n_tiles_y, 4). Each entry defines a tile
        as (x, y, width, height).
    """
    x_positions, widths = _compute_chunk_sizes_positions(dimensions[0], chunk_size[0], min_coord=min_coordinates[0])
    y_positions, heights = _compute_chunk_sizes_positions(dimensions[1], chunk_size[1], min_coord=min_coordinates[1])

    # Generate the tiles
    # x/width are inner list
    # y/height are outer list
    # in line with dask.array.block
    tiles = np.array(
        [
            [[x, y, width, height] for x, width in zip(x_positions, widths, strict=True)]
            for y, height in zip(y_positions, heights, strict=True)
        ],
        dtype=int,
    )
    return tiles


def _read_chunks(
    func: Callable[..., NDArray], slide: Any, coords: NDArray, n_channel: int, dtype: np.dtype, **func_kwargs: Any
) -> list[list[NDArray]]:
    """Abstract factory method to tile a large microscopy image.

    Parameters
    ----------
    func
        Function to retrieve a rectangular tile from the slide image. Must take the
        arguments:

            - slide Full slide image
            - x0: x (col) coordinate of upper left corner of chunk
            - y0: y (row) coordinate of upper left corner of chunk
            - width: Width of chunk
            - height: Height of chunk

        and should return the chunk as numpy array of shape (c, y, x)
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
    # Inner list becomes dim=-1 (x in cyx)
    # Outer list becomes dim=-2 (y in cyx)
    # see dask.array.block
    chunks = [
        [
            da.from_delayed(
                delayed(func)(
                    slide,
                    x0=coords[tile_y, tile_x, 0],
                    y0=coords[tile_y, tile_x, 1],
                    width=coords[tile_y, tile_x, 2],
                    height=coords[tile_y, tile_x, 3],
                    **func_kwargs,
                ),
                dtype=dtype,
                shape=(n_channel, *coords[tile_y, tile_x, [3, 2]]),
            )
            for tile_x in range(coords.shape[1])
        ]
        for tile_y in range(coords.shape[0])
    ]
    return chunks


def _assemble(chunks: list[list[NDArray]]) -> NDArray:
    """Assemble chunks (delayed)"""
    return da.block(chunks, allow_unknown_chunksizes=True)
