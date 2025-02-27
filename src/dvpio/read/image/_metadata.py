from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel


def _get_value_from_nested_dict(nested_dict: dict, keys: list, default_return_value: Any = None) -> Any:
    """Get a specific value from a nested dictionary"""
    for key in keys[:-1]:
        if not isinstance(nested_dict, dict):
            raise ValueError(f"Returned type of key {key} in nested dict is not expected dict but {type(dict)}")
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
        This value does not consider the magnifications by additional optical elements
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

    @classmethod
    @abstractmethod
    def from_file(path: str):
        """Load metadata from microscopy image in path"""
        ...


class CZIImageMetadata(ImageMetadata):
    metadata: dict[str, Any]

    # *_PATH keys in nested dict that lead to the metadata field
    CHANNEL_INFO_PATH: ClassVar = (
        "ImageDocument",
        "Metadata",
        "Information",
        "Image",
        "Dimensions",
        "Channels",
        "Channel",
    )
    MPP_PATH: ClassVar = ("ImageDocument", "Metadata", "Scaling", "Items", "Distance")
    OBJECTIVE_NAME_PATH: ClassVar = ("ImageDocument", "Metadata", "Scaling", "AutoScaling", "ObjectiveName")
    OBJECTIVE_NOMINAL_MAGNIFICATION_PATH: ClassVar = (
        "ImageDocument",
        "Metadata",
        "Information",
        "Instrument",
        "Objectives",
        "Objective",
    )

    @property
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
        channels = _get_value_from_nested_dict(self.metadata, self.CHANNEL_INFO_PATH, default_return_value=[])

        # For a single channel, a dict is returned
        if isinstance(channels, dict):
            channels = [channels]

        return channels

    @property
    def channel_id(self) -> list[int]:
        """Parse channel metadata to list of channel ids

        Notes
        -----
        Per channel, IDs are stored under the key `@Id` in the form `Channel:<channel id>`
        in the channel metadata
        """
        return [self._parse_channel_id(channel.get("@Id")) for channel in self._channel_info]

    @property
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
        return _get_value_from_nested_dict(self.metadata, self.MPP_PATH, [])

    @property
    def mpp_x(self) -> float | None:
        """Return resolution in X dimension in [meters per pixel]"""
        return self._parse_mpp_dim(self._mpp, dimension="X")

    @property
    def mpp_y(self) -> float | None:
        """Resolution in Y dimension in [meters per pixel]"""
        return self._parse_mpp_dim(self._mpp, dimension="Y")

    @property
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
            nested_dict=self.metadata, keys=self.OBJECTIVE_NAME_PATH, default_return_value=None
        )

    @property
    def objective_nominal_magnification(self) -> float | None:
        """Utilized objective_nominal_magnification

        Note
        ----
        Given the utilized objective the utilized objective_nominal_magnification can be extracted
        from the metadata on all available Objectives. The objective_nominal_magnification of an objective
        is given as `NominalMagnification` field.
        """
        objectives = _get_value_from_nested_dict(
            self.metadata, keys=self.OBJECTIVE_NOMINAL_MAGNIFICATION_PATH, default_return_value=[]
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
        """Parse metadata from file path

        Parameters
        ----------
        path
            Path to `.czi` file.

        Returns
        -------
        Parsed metadata as pydantic model
        """
        from pylibCZIrw.czi import open_czi

        with open_czi(path) as czi:
            metadata = czi.metadata

        return cls(metadata=metadata)
