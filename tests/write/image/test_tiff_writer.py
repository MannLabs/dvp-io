from typing import Any

import dask.array as da
import numpy as np
import pytest
import spatialdata as sd
import xarray as xr

from dvpio.write.image.tiff_writer import get_raster


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
