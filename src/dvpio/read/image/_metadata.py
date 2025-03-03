from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal
from warnings import warn

import openslide
from pydantic import BaseModel
from pylibCZIrw.czi import open_czi

from dvpio._utils import is_parsed


def _get_value_from_nested_dict(nested_dict: dict, keys: list, default_return_value: Any = None) -> Any:
    """Get a specific value from a nested dictionary"""
    for key in keys[:-1]:
        if not isinstance(nested_dict, dict):
            raise ValueError(f"Returned type of key {key} in nested dict is not expected dict but {type(nested_dict)}")
        nested_dict = nested_dict.get(key, {})

    return nested_dict.get(keys[-1], default_return_value)


class ImageMetadata(BaseModel, ABC):
    metadata: dict[str, dict | list | str]

    @property
    @abstractmethod
    def objective_nominal_magnification(self) -> int | None:
        """Nominal magnification of the microscope objective

        Note
        ----
        This value does not consider the magnification by additional optical elements
        in the specific microscopy setup
        """
        ...

    @property
    @abstractmethod
    def mpp_x(self) -> float:
        """Resolution of the image in meters per pixel along x-axis"""
        ...

    @property
    @abstractmethod
    def mpp_y(self) -> float:
        """Resolution of the image in meters per pixel along y-axis"""
        ...

    @property
    @abstractmethod
    def mpp_z(self) -> float:
        """Resolution of the image in meters per pixel along z-axis"""
        ...

    @property
    @abstractmethod
    def channel_names(self) -> list[str] | None:
        """Names of the microscopy channels"""
        ...

    @property
    @abstractmethod
    def image_type(self) -> str:
        """Indicator of the original image format/microscopy vendor"""
        ...

    @property
    def parsed_properties(self) -> dict[str, Any]:
        """Return a dictionary of all parsed metadata fields marked with the `_is_parsed` attribute"""
        return {
            attr: getattr(self, attr)
            for attr in dir(self.__class__)
            if isinstance(getattr(self.__class__, attr), property)
            and getattr(getattr(self.__class__, attr).fget, "_is_parsed", False)
        }

    @classmethod
    @abstractmethod
    def from_file(cls, path: str) -> BaseModel:
        """Parse metadata from file path

        Parameters
        ----------
        path
            Path to microscopy file.

        Returns
        -------
        Parsed metadata as pydantic model
        """
        ...


class CZIImageMetadata(ImageMetadata):
    metadata: dict[str, Any]

    # *_PATH keys in nested dict that lead to the metadata field
    _CHANNEL_INFO_PATH: ClassVar = (
        "ImageDocument",
        "Metadata",
        "Information",
        "Image",
        "Dimensions",
        "Channels",
        "Channel",
    )
    _MPP_PATH: ClassVar = ("ImageDocument", "Metadata", "Scaling", "Items", "Distance")
    _OBJECTIVE_NAME_PATH: ClassVar = ("ImageDocument", "Metadata", "Scaling", "AutoScaling", "ObjectiveName")
    _OBJECTIVE_NOMINAL_MAGNIFICATION_PATH: ClassVar = (
        "ImageDocument",
        "Metadata",
        "Information",
        "Instrument",
        "Objectives",
        "Objective",
    )

    @property
    @is_parsed
    def image_type(self) -> str:
        return "czi"

    def _parse_channel_id(self, channel_name: str) -> int:
        """Parse CZI channel id representation to channel index"""
        if channel_name is None:
            return
        return int(channel_name.replace("Channel:", ""))

    def _parse_mpp_dim(self, mpp: list[dict[str, str]], dimension: str) -> float | None:
        """Parse the pixel resolution entry in CZI metadata

        Note
        ----
        Per dimension, the resolution is stored as dict with the keys @Id (X/Y/Z),
        and optional `Value` key (resolution as float in meters per pixel).
        """
        entry = next((e for e in mpp if e.get("@Id") == dimension), {})
        mpp_dim = entry.get("Value", None)

        return float(mpp_dim) if mpp_dim else None

    @property
    def _channel_info(self) -> list[dict[str, str]]:
        """Obtain channel metadata from CZI metadata file

        Notes
        -----
        CZI represents strings in the `Channel` metadata field as list of dicts.
        The dict minimally contains an `@ID` and a `PixelType` key, but
        may also contain a `Name` key.
        """
        channels = _get_value_from_nested_dict(self.metadata, self._CHANNEL_INFO_PATH, default_return_value=[])

        # For a single channel, a dict is returned
        if isinstance(channels, dict):
            channels = [channels]

        return channels

    @property
    @is_parsed
    def channel_id(self) -> list[int]:
        """Parse channel metadata to list of channel ids

        Notes
        -----
        Per channel, IDs are stored under the key `@Id` in the form `Channel:<channel id>`
        in the channel metadata
        """
        return [self._parse_channel_id(channel.get("@Id")) for channel in self._channel_info]

    @property
    @is_parsed
    def channel_names(self) -> list[str]:
        """Parse channel metadata to list of channel ids

        Returns
        -------
        List of channel names
            If no channel name is given, falls back to returning index of channel as string

        Notes
        -----
        Per channel, names are stored under the key `@Name` as str
        in the channel metadata
        """
        return [channel.get("@Name", str(idx)) for idx, channel in enumerate(self._channel_info)]

    @property
    def _mpp(self) -> dict[str, dict[str, str]]:
        """Parse pixel resolution from slide image

        Note
        ----
        Pixel resolution is stored in `Distance` field and always specified in meters per pixel
        """
        return _get_value_from_nested_dict(self.metadata, self._MPP_PATH, [])

    @property
    @is_parsed
    def mpp_x(self) -> float | None:
        """Return resolution in X dimension in [meters per pixel]"""
        return self._parse_mpp_dim(self._mpp, dimension="X")

    @property
    @is_parsed
    def mpp_y(self) -> float | None:
        """Resolution in Y dimension in [meters per pixel]"""
        return self._parse_mpp_dim(self._mpp, dimension="Y")

    @property
    @is_parsed
    def mpp_z(self) -> float | None:
        """Resolution in Z dimension in [meters per pixel]"""
        return self._parse_mpp_dim(self._mpp, dimension="Z")

    @property
    def objective_name(self) -> str | None:
        """Utilized objective name. Required to infer objective_nominal_magnification

        Note
        ----
        Objective Name is stored as string in `ObjectiveName` field. Presumably,
        this represents the currently utilized objective
        """
        return _get_value_from_nested_dict(
            nested_dict=self.metadata, keys=self._OBJECTIVE_NAME_PATH, default_return_value=None
        )

    @property
    @is_parsed
    def objective_nominal_magnification(self) -> float | None:
        """Utilized objective_nominal_magnification

        Note
        ----
        Given the utilized objective the utilized objective_nominal_magnification can be extracted
        from the metadata on all available Objectives. The objective_nominal_magnification of an objective
        is given as `NominalMagnification` field.
        """
        objectives = _get_value_from_nested_dict(
            self.metadata, keys=self._OBJECTIVE_NOMINAL_MAGNIFICATION_PATH, default_return_value=[]
        )

        if isinstance(objectives, dict):
            objectives = [objectives]
        objective_nominal_magnification = None
        for objective in objectives:
            if objective.get("@Name") == self.objective_name:
                objective_nominal_magnification = objective.get("NominalMagnification")
        return float(objective_nominal_magnification) if objective_nominal_magnification else None

    @classmethod
    def from_file(cls, path: str) -> BaseModel:
        with open_czi(path) as czi:
            metadata = czi.metadata

        return cls(metadata=metadata)


class OpenslideImageMetadata(ImageMetadata):
    metadata: dict[str, Any]

    # Openslide returns MPP in micrometers per pixel
    # Convert it to meters to pixel for compatibility reasons
    # See https://openslide.org/api/python/#standard-properties
    _MICROMETER_TO_METER_CONVERSION: ClassVar[float] = 1e-6

    # Openslide always returns RGBA images. Set channel ids + names as constants
    _CHANNEL_IDS: ClassVar[list[int]] = [0, 1, 2, 3]
    _CHANNEL_NAMES: ClassVar[list[str]] = ["R", "G", "B", "A"]

    @property
    @is_parsed
    def image_type(self) -> str:
        """Indicator of the original image format/microscopy vendor, defaults to openslide if unknown."""
        return self.metadata.get(openslide.PROPERTY_NAME_VENDOR, "openslide")

    @property
    @is_parsed
    def objective_nominal_magnification(self) -> float | None:
        magnification = self.metadata.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
        return float(magnification) if magnification is not None else None

    @property
    @is_parsed
    def channel_id(self) -> list[int]:
        # Openslide returns RGBA images (4 channels)
        # https://openslide.org/api/python/#openslide.OpenSlide.read_region
        return self._CHANNEL_IDS

    @property
    @is_parsed
    def channel_names(self) -> list[int]:
        # Openslide returns RGBA images (channels R, G, B, A)
        # https://openslide.org/api/python/#openslide.OpenSlide.read_region
        return self._CHANNEL_NAMES

    @property
    @is_parsed
    def mpp_x(self) -> float | None:
        mpp_x = self.metadata.get(openslide.PROPERTY_NAME_MPP_X)
        return self._MICROMETER_TO_METER_CONVERSION * float(mpp_x) if mpp_x is not None else None

    @property
    @is_parsed
    def mpp_y(self) -> float | None:
        mpp_y = self.metadata.get(openslide.PROPERTY_NAME_MPP_Y)
        return self._MICROMETER_TO_METER_CONVERSION * float(mpp_y) if mpp_y is not None else None

    @property
    @is_parsed
    def mpp_z(self) -> None:
        warn(
            "Whole Slide images read by openslide do not contain a MPP property in Z dimension, return None",
            stacklevel=1,
        )
        return

    @classmethod
    @is_parsed
    def from_file(cls, path) -> BaseModel:
        slide = openslide.OpenSlide(path)
        return cls(metadata=slide.properties)


def read_metadata(path: str, image_type: Literal["czi", "openslide"], parse_metadata: bool = True) -> dict[str, Any]:
    """Parse relevant microscopy metadata of dvp-io supported image file

    Currently only supports `czi` files and `openslide`-compatible files.

    Parameters
    ----------
    path
        Path to image file
    reader_type
        One of the supported image data types (`czi`, `openslide`)
    parse_metadata
        Whether to extract relevant metadata or return the raw metadata as json-style dictionary

    Returns
    -------
    Metadata as dictionary

    If `parse_metadata` is true, returns a dict with the following keys and associated values

        - image_type: str | None
            Name of original type: czi for Carl Zeiss, vendor name for openslide
        - objective_nominal_magnification: float | None
            Nominal magnification of objective, not considering additional optical setups
        - mpp_x: float | None
            Resolution in `meters per pixel` in x-dimension
        - mpp_x: float | None
            Resolution in `meters per pixel` in y-dimension
        - mpp_x: float | None
            Resolution in `meters per pixel` in z-dimension
        - channel_id: list[int] | None
            List of indices of microscopy channels
        - channel_names: list[str] | None
            List of channel names

    Example
    -------

    .. code-block:: python

        import spatialdata as sd
        from dvpio.read.image import read_czi, parse_metadata

        img_path = "./data/kabatnik2023_20211129_C1.czi"

        # Initialize spatialdata
        sdata = sd.SpatialData()

        # Assign image
        sdata.images["image"] = read_czi(img_path)

        # Get controlled attributes from metadata
        image_metadata = parse_metadata(img_path, image_type="czi", parse_metadata=True)
        image_metadata
        > {
            'channel_id': [0],
            'channel_names': ['TL Brightfield'],
            'image_type': 'czi',
            'mpp_x': 2.1999999999999998e-07,
            'mpp_y': 2.1999999999999998e-07,
            'mpp_z': 1.5e-06,
            'objective_nominal_magnification': 20.0
        }

        # Get the full metadata document
        image_metadata = parse_metadata(img_path, image_type="czi", parse_metadata=False)
        > {
        'ImageDocument':
            {'Metadata':
                ...
                }
            ...
            }
        ...
        }

        # Assign it to spatialdata.SpatialData.attrs slot for future reference
        # It is recommended to use the same name as the image
        sdata.attrs["metadata"] = {
            "image": image_metadata
        }

        # Write
        # sdata.write("/path/to/sdata.zarr")

    """
    if image_type == "czi":
        metadata = CZIImageMetadata.from_file(path)
    elif image_type == "openslide":
        metadata = OpenslideImageMetadata.from_file(path)
    else:
        raise NotImplementedError(
            "Parameter image_type needs to be `czi` or `openslide` for automated metadata parsing"
        )

    if parse_metadata:
        return metadata.parsed_properties

    return metadata.metadata
