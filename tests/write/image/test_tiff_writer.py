from pathlib import Path
from typing import Any, Literal

import dask.array as da
import numpy as np
import pytest
import spatialdata as sd
import tifffile as tiff
import xarray as xr

from dvpio.write.image.tiff_writer import _is_rgb, _iter_tiles, get_raster, write_ome_tiff


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
    multiscale_image = sdata.images["blobs_multiscale_image"]
    assert all(scale in multiscale_image.children for scale in ("scale0", "scale1", "scale2"))
    return multiscale_image


class TestGetRaster:
    def test_get_raster__dataarray(self, image: xr.DataTree) -> None:
        """Test that get_raster returns a data array"""
        result = get_raster(image)
        assert isinstance(result, xr.DataArray)

    @pytest.mark.parametrize("level", ["scale1", "scale2", "scale3", None])
    def test_get_raster__datatree(self, image: xr.DataTree, level: str) -> None:
        """Test that get_raster returns a data array"""
        result = get_raster(image, level=level)
        assert isinstance(result, xr.DataArray)

    @pytest.mark.parametrize("image", [np.zeros(shape=(512, 512)), da.zeros(shape=(512, 512))])
    def test_get_raster__raises_value_error(self, image: Any) -> None:
        """Test that get_raster raisese for unknown data types"""
        with pytest.raises(ValueError):
            _ = get_raster(image)

    @pytest.mark.parametrize("key", "non-existent")
    def test_get_raster__raises_key_error(self, multiscale_image: Any, key: str) -> None:
        """Test that get_raster raisese for unknown data types"""
        with pytest.raises(KeyError):
            _ = get_raster(multiscale_image, level=key)


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


class TestIsRGB:
    @pytest.fixture(
        params=[
            {"channel_names": ["r", "g", "b"]},
            {"channel_names": ["R", "G", "B"]},
            {"channel_names": ["red", "green", "blue"]},
            {"channel_names": ["Red", "Green", "Blue"]},
        ]
    )
    def rgb_image(self, image: sd.models.Image2DModel, request) -> sd.models.Image2DModel:
        """Create an rgb image from a spatialdata image model"""
        dim_shapes = dict(zip(image.dims, image.shape, strict=True))
        assert dim_shapes["c"] == 3

        image_rgb = image.copy()

        return image_rgb.assign_coords(coords={"c": request.param["channel_names"]})

    def test__is_rgb__rgb_image(self, rgb_image: sd.models.Image2DModel) -> None:
        """Test that _check_is_rgb returns True if RGB image is passed"""
        assert _is_rgb(rgb_image)

    def test__is_rgb__grayscale_image(self, image: sd.models.Image2DModel) -> None:
        """Test that _check_is_rgb returns True if RGB image is passed"""
        assert not _is_rgb(image)


class TestWriteOmeTiff:
    @pytest.fixture
    def image_path(self, tmp_path) -> Path:
        return tmp_path / "image.tiff"

    @pytest.fixture
    def image_3channel(self, image) -> Path:
        # Validate that there are 3 channels
        dim_shapes = dict(zip(image.dims, image.shape, strict=True))
        assert dim_shapes["c"] == 3
        assert sd.models.get_channel_names(image) == [0, 1, 2]
        return image.copy()

    @pytest.mark.parametrize("tile_shape", [(256, 256), (512, 512), (1024, 1024)])
    def test_write_ome_tiff__dataarray(
        self, image_path, image: sd.models.Image2DModel, tile_shape: tuple[int, int]
    ) -> None:
        write_ome_tiff(image_path, image, tile_shape=tile_shape)

        new_image = tiff.imread(image_path)

        assert np.array_equal(new_image, image.data.compute())

    @pytest.mark.parametrize("tile_shape", [(256, 256), (512, 512), (1024, 1024)])
    @pytest.mark.parametrize("level", ["scale0", "scale1", "scale2", None])
    def test_write_ome_tiff__datatree(
        self, image_path, multiscale_image: sd.models.Image2DModel, level: str, tile_shape: tuple[int, int]
    ) -> None:
        write_ome_tiff(image_path, multiscale_image, level=level, tile_shape=tile_shape)

        new_image = tiff.imread(image_path)

        # Get image at correct resolution
        ref_image_multiscale = get_raster(multiscale_image, level=level)
        ref_image = ref_image_multiscale.data.compute()

        assert np.array_equal(new_image, ref_image)

    @pytest.mark.parametrize("rgb", [True, False, None])
    def test_write_ome_tiff__test_rgb(
        self, image_path, image_3channel: sd.models.Image2DModel, rgb: bool | None
    ) -> None:
        write_ome_tiff(image_path, image_3channel, rgb=rgb)

        with tiff.TiffFile(image_path, mode="r") as img:
            photometric_type = img.pages[0].photometric.name.lower()

        if rgb is True:
            assert photometric_type == "rgb"
        else:
            assert photometric_type == "minisblack"
