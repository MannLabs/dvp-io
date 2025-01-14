import pytest
from tifffile import imread as tiffread

from dvpio.read.image import read_custom


@pytest.mark.parametrize(["filename"], [["./data/blobs/blobs/images/binary-blobs.tiff"]])
def test_custom(filename: str) -> None:
    img = read_custom(filename, dims=("c", "y", "x"))

    img_groundtruth = tiffread(filename)

    assert img.shape == img_groundtruth.shape
    assert (img == img_groundtruth).all()
