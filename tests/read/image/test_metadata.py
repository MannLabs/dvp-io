from typing import Any

import pytest
from pydantic import BaseModel

from dvpio.read.image._metadata import CZIImageMetadata, OpenslideImageMetadata, _get_value_from_nested_dict

CZI_GROUND_TRUTH = {
    "./data/zeiss/zeiss/rect-upper-left.czi": {
        "channel_ids": [0],
        "channel_names": ["0"],
        "mpp_x": 0,
        "mpp_y": 0,
        "mpp_z": 0,
        "objective_nominal_magnification": None,
    },
    "./data/zeiss/zeiss/rect-upper-left.multi-channel.czi": {
        "channel_ids": [0, 1, 2],
        "channel_names": ["0", "1", "2"],
        "mpp_x": 0,
        "mpp_y": 0,
        "mpp_z": 0,
        "objective_nominal_magnification": None,
    },
    "./data/zeiss/zeiss/rect-upper-left.rgb.czi": {
        "channel_ids": [0],
        "channel_names": ["0"],
        "mpp_x": 0,
        "mpp_y": 0,
        "mpp_z": 0,
        "objective_nominal_magnification": None,
    },
    "./data/zeiss/zeiss/zeiss_multi-channel.czi": {
        "channel_ids": [0, 1],
        "channel_names": ["DAPI", "PGC"],
        "mpp_x": 4.5502152331985306e-07,
        "mpp_y": 4.5502152331985306e-07,
        "mpp_z": None,
        "objective_nominal_magnification": 5,
    },
    "./data/zeiss/zeiss/kabatnik2023_20211129_C1.czi": {
        "channel_ids": [0],
        "channel_names": ["TL Brightfield"],
        "mpp_x": 2.1999999999999998e-07,
        "mpp_y": 2.1999999999999998e-07,
        "mpp_z": 1.5e-06,
        "objective_nominal_magnification": 20,
    },
}

OPENSLIDE_GROUND_TRUTH = {
    "./data/openslide-mirax/Miarx2.2-4-PNG.mrxs": {
        "channel_ids": [0, 1, 2, 3],
        "channel_names": ["R", "G", "B", "A"],
        "mpp_x": 0.23387573964496999 * 1e-6,
        "mpp_y": 0.234330708661417 * 1e-6,
        "mpp_z": None,
        "objective_nominal_magnification": 20,
    }
}


@pytest.fixture(scope="module")
def nested_dict():
    nd = {"A": [], "B": {"C": "c"}}
    return nd


@pytest.mark.parametrize((["keys", "output"]), [(["A"], []), (["B", "C"], "c"), (["E"], None), (["B", "D"], None)])
def test_get_value_from_nested_dict(nested_dict: dict[str, Any], keys: list[str], output: str | None) -> None:
    assert _get_value_from_nested_dict(nested_dict, keys=keys, default_return_value=None) == output


@pytest.mark.parametrize((["keys", "default_return_value"]), [("E", []), ("E", {}), (["B", "D"], []), (["B", "D"], {})])
def test_get_value_from_nested_dict_return_value(
    nested_dict: dict[str, Any], keys: list[str], default_return_value: str | None
) -> None:
    """Test default return value on keys that do not match any fields in the nested dict"""
    assert (
        _get_value_from_nested_dict(nested_dict, keys=keys, default_return_value=default_return_value)
        == default_return_value
    )


@pytest.fixture(params=CZI_GROUND_TRUTH.keys())
def czi_metadata_parser(request) -> BaseModel:
    path = request.param
    return (CZIImageMetadata.from_file(path), CZI_GROUND_TRUTH[path])


@pytest.fixture(params=OPENSLIDE_GROUND_TRUTH.keys())
def openslide_metadata_parser(request) -> BaseModel:
    path = request.param
    return (OpenslideImageMetadata.from_file(path), OPENSLIDE_GROUND_TRUTH[path])


def test_czi_channel_id_parser(czi_metadata_parser):
    metadata, ground_truth = czi_metadata_parser
    assert metadata.channel_id == ground_truth["channel_ids"]


def test_czi_image_type_parser(czi_metadata_parser):
    metadata, ground_truth = czi_metadata_parser
    assert metadata.image_type == "czi"


def test_czi_channel_names_parser(czi_metadata_parser):
    metadata, ground_truth = czi_metadata_parser
    assert metadata.channel_names == ground_truth["channel_names"]


def test_czi_mpp_parser(czi_metadata_parser):
    metadata, ground_truth = czi_metadata_parser
    assert metadata.mpp_x == ground_truth["mpp_x"]
    assert metadata.mpp_y == ground_truth["mpp_y"]
    assert metadata.mpp_z == ground_truth["mpp_z"]


def test_czi_magnification_parser(czi_metadata_parser):
    metadata, ground_truth = czi_metadata_parser
    assert metadata.objective_nominal_magnification == ground_truth["objective_nominal_magnification"]
    assert metadata.mpp_y == ground_truth["mpp_y"]
    assert metadata.mpp_z == ground_truth["mpp_z"]
