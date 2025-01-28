import pytest
from tifffile import imread as tiffread

from dvpio.read.image import read_custom


@pytest.mark.parametrize(
    ["filename"], [["./data/blobs/blobs/images/binary-blobs.tiff"], ["./data/blobs/blobs/images/binary-blobs*.tiff"]]
)
def test_custom(filename: str) -> None:
    # TODO Create a more elegant solution
    img = read_custom(filename, imread=lambda path: tiffread(path).squeeze(), dims=("c", "y", "x"))

    img_groundtruth = tiffread(filename)

    assert img.shape == img_groundtruth.shape
    assert (img == img_groundtruth).all()
