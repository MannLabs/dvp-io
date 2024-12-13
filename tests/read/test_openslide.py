import numpy as np
import openslide
import pytest
from spatialdata_io.experimental.openslide import read_openslide


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
