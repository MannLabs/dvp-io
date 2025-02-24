from typing import Any

import dask.array as da
import numpy as np
import pytest
from dask import delayed
from numpy.typing import NDArray

from dvpio.read.image._utils import _compute_chunks, _read_chunks


@pytest.mark.parametrize(
    ("dimensions", "tile_size", "min_coordinates", "result"),
    [
        # Regular grid 2x2
        (
            (2, 2),
            (1, 1),
            (0, 0),
            np.array([[[0, 0, 1, 1], [0, 1, 1, 1]], [[1, 0, 1, 1], [1, 1, 1, 1]]]),
        ),
        # Different tile sizes
        (
            (3, 3),
            (2, 2),
            (0, 0),
            np.array([[[0, 0, 2, 2], [0, 2, 2, 1]], [[2, 0, 1, 2], [2, 2, 1, 1]]]),
        ),
        # Negative tile start
        (
            (2, 2),
            (1, 1),
            (-1, 0),
            np.array([[[-1, 0, 1, 1], [-1, 1, 1, 1]], [[0, 0, 1, 1], [0, 1, 1, 1]]]),
        ),
    ],
)
def test_compute_chunks(
    dimensions: tuple[int, int],
    tile_size: tuple[int, int],
    min_coordinates: tuple[int, int],
    result: NDArray,
) -> None:
    tiles = _compute_chunks(dimensions=dimensions, tile_size=tile_size, min_coordinates=min_coordinates)

    assert (tiles == result).all()


@pytest.mark.parametrize(
    ("dimensions", "tile_size", "min_coordinates"),
    [
        # Regular grid 2x2
        (
            (2, 2),
            (1, 1),
            (0, 0),
        ),
        # Different tile sizes
        (
            (3, 3),
            (2, 2),
            (0, 0),
        ),
        # Negative tile start
        (
            (2, 2),
            (1, 1),
            (-1, 0),
        ),
    ],
)
def test_read_chunks(
    dimensions: tuple[int, int],
    tile_size: tuple[int, int],
    min_coordinates: tuple[int, int],
) -> None:
    """Test if tiles can be assembled to dask array"""

    @delayed
    def func(slide: Any, coords: Any, size: tuple[int]) -> NDArray[np.int_]:
        """Create arrays in shape of tiles"""
        return da.zeros(shape=size)

    coords = _compute_chunks(dimensions=dimensions, tile_size=tile_size, min_coordinates=min_coordinates)

    tiles_ = _read_chunks(func, slide=None, coords=coords, n_channel=1, dtype=np.uint8)
    tiles = da.block(tiles_)

    assert tiles.shape == (1, *dimensions)


@pytest.mark.parametrize(("dtype"), [(np.uint8), (np.int16), (np.float32)])
def test_read_chunks_dtype(dtype) -> None:
    """Test if tiles can be assembled to dask array"""

    @delayed
    def func(slide: Any, coords: Any, size: tuple[int]) -> NDArray[np.int_]:
        """Create arrays in shape of tiles"""
        return da.zeros(shape=size)

    coords = _compute_chunks(dimensions=(2, 2), tile_size=(1, 1), min_coordinates=(0, 0))

    tiles_ = _read_chunks(func, slide=None, coords=coords, n_channel=1, dtype=dtype)
    tiles = da.block(tiles_)

    assert tiles.dtype == dtype
