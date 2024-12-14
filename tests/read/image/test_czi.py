import pytest
from pylibCZIrw import czi as pyczi

from dvpio.read.image import read_czi


@pytest.mark.parametrize(
    ("dataset", "xmin", "ymin", "width", "height"),
    [
        # RGB
        # ("./data/zeiss/zeiss/kabatnik2023_20211129_C1.czi", -170000, 40000, 10000, 10000),
        # Multichannel
        ("./data/zeiss/zeiss/zeiss_multi-channel.czi", 0, 0, 1000, 1000),
    ],
)
def test_read_czi(dataset: str, xmin: int, ymin: int, width: int, height: int) -> None:
    # Get reference with CZI reader
    czidoc_r = pyczi.CziReader(dataset)

    # Returns numpy array with shape (y, x, c)
    xmin_czi, ymin_czi, total_width, total_height = czidoc_r.total_bounding_rectangle
    img_ref = czidoc_r.read(plane={"C": 0, "T": 0, "Z": 0}, roi=(xmin, ymin, width, height))

    # Test function
    array = read_czi(dataset)

    # Coordinate systems are not aligned, modify roi
    x, y = xmin - xmin_czi, ymin - ymin_czi
    img_test = array[:, x : x + width, y : y + height].compute().transpose("x", "y", "c")

    assert array.shape[1:] == (total_width, total_height)
    assert (img_test == img_ref).all()


@pytest.mark.parametrize(
    ("dataset", "channels", "result_dim"),
    [
        ("./data/zeiss/zeiss/zeiss_multi-channel.czi", 0, 1),
        ("./data/zeiss/zeiss/zeiss_multi-channel.czi", [0], 1),
        ("./data/zeiss/zeiss/zeiss_multi-channel.czi", [0, 1], 2),
    ],
)
def test_read_czi_multichannel(
    dataset: str,
    channels: int | list[int],
    result_dim: int,
) -> None:
    # Get reference with CZI reader
    czidoc_r = pyczi.CziReader(dataset)

    # Returns numpy array with shape (y, x, c)
    xmin_czi, ymin_czi, total_width, total_height = czidoc_r.total_bounding_rectangle

    # Test function
    array = read_czi(dataset, channels=channels)

    assert array.shape == (result_dim, total_width, total_height)
