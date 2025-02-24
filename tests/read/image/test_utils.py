from typing import Any

import dask.array as da
import numpy as np
import pytest
from dask import delayed
from numpy.typing import NDArray

from dvpio.read.image._utils import _compute_chunk_sizes_positions, _compute_chunks, _read_chunks


@pytest.mark.parametrize(
    ("size", "chunk", "min_coordinate", "positions", "lengths"),
    [
        # Evenly spaced subsets
        (3, 1, 0, np.array([0, 1, 2]), np.array([1, 1, 1])),
        # Unevenly spaced subsets
        (3, 2, 0, np.array([0, 2]), np.array([2, 1])),
        # Negative start coordinate
        (3, 1, -1, np.array([-1, 0, 1]), np.array([1, 1, 1])),
        # Uneven spacing and negative start coordinate
        (3, 2, -1, np.array([-1, 1]), np.array([2, 1])),
    ],
)
def test_compute_chunk_sizes_positions(
    size: int,
    chunk: int,
    min_coordinate: int,
    positions: NDArray[np.number],
    lengths: NDArray[np.number],
) -> None:
    """Test whether 1D chunking"""
    computed_positions, computed_lengths = _compute_chunk_sizes_positions(size, chunk, min_coordinate)
    assert (positions == computed_positions).all()
    assert (lengths == computed_lengths).all()


@pytest.mark.parametrize(
    ("dimensions", "chunk_size", "min_coordinates", "result"),
    [
        # Regular grid 2x2
        (
            (2, 2),
            (1, 1),
            (0, 0),
            np.array([[[0, 0, 1, 1], [1, 0, 1, 1]], [[0, 1, 1, 1], [1, 1, 1, 1]]]),
        ),
        # Different tile sizes
        (
            (3, 3),
            (2, 2),
            (0, 0),
            np.array([[[0, 0, 2, 2], [2, 0, 1, 2]], [[0, 2, 2, 1], [2, 2, 1, 1]]]),
        ),
        # Negative tile start
        (
            (2, 2),
            (1, 1),
            (-1, 0),
            np.array([[[-1, 0, 1, 1], [0, 0, 1, 1]], [[-1, 1, 1, 1], [0, 1, 1, 1]]]),
        ),
    ],
)
def test_compute_chunks(
    dimensions: tuple[int, int],
    chunk_size: tuple[int, int],
    min_coordinates: tuple[int, int],
    result: NDArray,
) -> None:
    """Test two dimensional chunking"""
    tiles = _compute_chunks(dimensions=dimensions, chunk_size=chunk_size, min_coordinates=min_coordinates)

    assert (tiles == result).all()


@pytest.mark.parametrize(
    ("dimensions", "chunk_size", "min_coordinates"),
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
        # Different sizes in x/y direction
        (
            (4, 3),
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
    chunk_size: tuple[int, int],
    min_coordinates: tuple[int, int],
) -> None:
    """Test if tiles can be assembled to dask array"""

    @delayed
    def func(slide: Any, coords: Any, size: tuple[int]) -> NDArray[np.int_]:
        """Create arrays in shape of tiles"""
        return da.zeros(shape=size)

    coords = _compute_chunks(dimensions=dimensions, chunk_size=chunk_size, min_coordinates=min_coordinates)

    tiles_ = _read_chunks(func, slide=None, coords=coords, n_channel=1, dtype=np.uint8)
    tiles = da.block(tiles_)

    assert tiles.shape == (1, *dimensions[::-1])


@pytest.mark.parametrize(("dtype"), [(np.uint8), (np.int16), (np.float32)])
def test_read_chunks_dtype(dtype) -> None:
    """Test if dtype of re-assembled tiles is equivalent to the original dtype"""

    @delayed
    def func(slide: Any, coords: Any, size: tuple[int]) -> NDArray[np.int_]:
        """Create arrays in shape of tiles"""
        return da.zeros(shape=size)

    coords = _compute_chunks(dimensions=(2, 2), chunk_size=(1, 1), min_coordinates=(0, 0))

    tiles_ = _read_chunks(func, slide=None, coords=coords, n_channel=1, dtype=dtype)
    tiles = da.block(tiles_)

    assert tiles.dtype == dtype
