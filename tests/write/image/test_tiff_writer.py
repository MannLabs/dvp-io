from typing import Any, Literal

import dask.array as da
import numpy as np
import pytest
import spatialdata as sd
import xarray as xr

from dvpio.write.image.tiff_writer import _iter_tiles, get_raster


@pytest.fixture
def sdata() -> sd.SpatialData:
    sdata = sd.datasets.blobs(length=513)
    assert "blobs_image" in sdata
    assert "blobs_multiscale_image" in sdata
    return sdata


@pytest.fixture
def image(sdata: sd.SpatialData) -> xr.DataArray:
    return sdata.images["blobs_image"]


@pytest.fixture
def multiscale_image(sdata: sd.SpatialData) -> xr.DataTree:
    return sdata.images["blobs_multiscale_image"]


class TestGetRaster:
    def test_get_raster__dataarray(self, image: xr.DataTree) -> None:
        """Test that get_raster returns a data array"""
        result = get_raster(image)
        assert isinstance(result, xr.DataArray)

    @pytest.mark.parametrize("level", ["scale1", "scale2", "scale3"])
    def test_get_raster__datatree(self, image: xr.DataTree, level: str) -> None:
        """Test that get_raster returns a data array"""
        result = get_raster(image, level=level)
        assert isinstance(result, xr.DataArray)

    @pytest.mark.parametrize("image", [np.zeros(shape=(512, 512)), da.zeros(shape=(512, 512))])
    def test_get_raster__raises(self, image: Any) -> None:
        """Test that get_raster raisese for unknown data types"""
        with pytest.raises(ValueError):
            _ = get_raster(image)


class TestIterTilesGenerator:
    @pytest.mark.parametrize("array_type", ["dask", "numpy"])
    @pytest.mark.parametrize(
        ("array_shape", "resulting_shapes"),
        [
            # Tile: y, x
            # Tile order: y, x
            ((512, 512), [(512, 512)]),
            ((1024, 1024), [(512, 512), (512, 512), (512, 512), (512, 512)]),
            ((1023, 1024), [(512, 512), (512, 512), (511, 512), (511, 512)]),
            (
                # 3 channels x Single tile/channel
                (3, 1, 512),
                [
                    (1, 512),
                    (1, 512),
                    (1, 512),
                ],
            ),
            (
                # 2 Z-stacks x 3 channels x 1 tile/channel
                (2, 3, 1, 512),
                [
                    (1, 512),
                    (1, 512),
                    (1, 512),
                    (1, 512),
                    (1, 512),
                    (1, 512),
                ],
            ),
        ],
        ids=("1-chunk", "4-chunks", "4-chunks-asymmetric", "multiple-channels", "multiple-channels-z-stacks"),
    )
    def test__iter_tiles(
        self, array_shape: np.ndarray, resulting_shapes: list[tuple], array_type: Literal["dask", "numpy"]
    ) -> None:
        """Test that iter tiles returns tiles in the order expected by tifffile"""
        # Setup dummy image
        image = np.zeros(shape=array_shape)
        if array_type == "dask":
            image = da.array(image)
        elif array_type == "numpy":
            image = image
        else:
            raise ValueError("Unexpected image type")

        iterator = _iter_tiles(array=image, tile_shape=(512, 512))
        tiles = list(iterator)

        assert all(tile.shape == reference_shape for tile, reference_shape in zip(tiles, resulting_shapes, strict=True))
