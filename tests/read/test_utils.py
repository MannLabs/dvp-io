from typing import Any

import dask.array as da
import numpy as np
import pytest
from numpy.typing import NDArray
from spatialdata_io.experimental._utils import _chunk_factory, _create_tiles


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
def test_create_tiles(
    dimensions: tuple[int, int],
    tile_size: tuple[int, int],
    min_coordinates: tuple[int, int],
    result: NDArray,
) -> None:
    tiles = _create_tiles(dimensions=dimensions, tile_size=tile_size, min_coordinates=min_coordinates)

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
def test_chunk_factory(
    dimensions: tuple[int, int],
    tile_size: tuple[int, int],
    min_coordinates: tuple[int, int],
) -> None:
    """Test if tiles can be assembled to dask array"""

    def func(slide: Any, coords: Any, size: tuple[int]) -> NDArray[np.int_]:
        """Create arrays in shape of tiles"""
        return np.zeros(shape=size)

    coords = _create_tiles(dimensions=dimensions, tile_size=tile_size, min_coordinates=min_coordinates)

    tiles_ = _chunk_factory(func, slide=None, coords=coords).compute()
    tiles = da.block(tiles_)

    assert tiles.shape == dimensions
