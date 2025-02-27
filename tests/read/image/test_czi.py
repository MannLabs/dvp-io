import numpy as np
import pytest
from pylibCZIrw import czi as pyczi

from dvpio.read.image import read_czi


@pytest.mark.parametrize(
    ("dataset", "xmin", "ymin", "width", "height"),
    [
        # Artificial data: Single gray scale, Multi gray scale, BGR
        ("./data/zeiss/zeiss/rect-upper-left.czi", 0, 0, 10, 10),
        ("./data/zeiss/zeiss/rect-upper-left.multi-channel.czi", 0, 0, 10, 10),
        ("./data/zeiss/zeiss/rect-upper-left.rgb.czi", 0, 0, 10, 10),
        # Kabatnik et al (RGB)
        ("./data/zeiss/zeiss/kabatnik2023_20211129_C1.czi", -150080, 56320, 5000, 4000),
        # Zeiss example data (Grayscale)
        ("./data/zeiss/zeiss/zeiss_multi-channel.czi", 0, 0, 2752, 2208),
    ],
)
def test_read_czi(dataset: str, xmin: int, ymin: int, width: int, height: int) -> None:
    # Get reference with CZI reader
    czidoc_r = pyczi.CziReader(dataset)

    # CZI returns numpy array in (y, x, c) shape
    xmin_czi, ymin_czi, total_width, total_height = czidoc_r.total_bounding_rectangle
    img_ref = czidoc_r.read(plane={"C": 0, "T": 0, "Z": 0}, roi=(xmin, ymin, width, height))

    # Test function
    array = read_czi(dataset, channels=0)

    # # Coordinate systems are not aligned, modify roi so that its coordinate system starts at (0, 0)
    x, y = xmin - xmin_czi, ymin - ymin_czi
    img_test = array[:, y : y + height, x : x + width].transpose("y", "x", "c")

    assert array.shape[1:] == (total_height, total_width)
    assert (img_test == img_ref).all()


@pytest.mark.parametrize(
    ("dataset", "channels", "result_dim"),
    [
        ("./data/zeiss/zeiss/rect-upper-left.multi-channel.czi", 0, 1),
        ("./data/zeiss/zeiss/rect-upper-left.multi-channel.czi", [0], 1),
        ("./data/zeiss/zeiss/rect-upper-left.multi-channel.czi", [0, 1], 2),
        ("./data/zeiss/zeiss/zeiss_multi-channel.czi", [0, 1], 2),
    ],
)
def test_read_czi_multichannel(
    dataset: str,
    channels: int | list[int],
    result_dim: int,
) -> None:
    # Test function
    img_test = read_czi(dataset, channels=channels)

    # Get reference with CZI reader
    czidoc_r = pyczi.CziReader(dataset)
    _, _, total_width, total_height = czidoc_r.total_bounding_rectangle

    if isinstance(channels, int):
        channels = [channels]

    # Reader returns (y, x, c=1) array
    # Stack all channels
    img_ref = np.concatenate([czidoc_r.read(plane={"C": channel}) for channel in range(result_dim)], axis=-1)

    assert img_test.shape == (result_dim, total_height, total_width)
    assert (img_test.transpose("y", "x", "c") == img_ref).all()
