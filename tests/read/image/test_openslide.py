import numpy as np
import openslide
import pytest

from dvpio.read.image import read_openslide
from dvpio.read.image.openslide import _get_img


@pytest.mark.parametrize(
    ("dataset", "xmin", "ymin", "width", "height"),
    [
        # Asymmetric
        ("./data/openslide-mirax/Mirax2.2-4-PNG.mrxs", 0, 0, 500, 1000),
    ],
)
def test_get_image_openslide(dataset, xmin: int, ymin: int, width: int, height: int) -> None:
    slide = openslide.OpenSlide(dataset)
    ground_truth_shape = (4, height, width)

    img = _get_img(slide, x0=xmin, y0=ymin, width=width, height=height, level=0)

    assert all(img_dim == ref_dim for img_dim, ref_dim in zip(img.shape, ground_truth_shape, strict=True))


# @pytest.mark.skipif(sys.platform != "darwin", reason="Tests fail online due to limited resources")
@pytest.mark.parametrize(
    ("dataset", "xmin", "ymin", "xmax", "ymax"),
    [
        ("./data/openslide-mirax/Mirax2.2-4-PNG.mrxs", 0, 0, 1000, 1000),
        # Asymmetric
        ("./data/openslide-mirax/Mirax2.2-4-PNG.mrxs", 0, 0, 500, 1000),
    ],
)
def test_read_openslide(dataset: str, xmin: int, xmax: int, ymin: int, ymax: int) -> None:
    """Test whether image can be loaded"""
    image_model = read_openslide(dataset, pyramidal=True)

    # Get a subset of the image
    test_image = image_model.scale0.image[:, ymin:ymax, xmin:xmax].transpose("y", "x", "c").to_numpy()

    # Read image directly with openslide
    slide = openslide.OpenSlide(dataset)
    ref_image = np.array(slide.read_region((xmin, ymin), level=0, size=(xmax - xmin, ymax - ymin)))

    assert (test_image == ref_image).all()
