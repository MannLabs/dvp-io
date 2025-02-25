from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, computed_field


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
        raise NotImplementedError

    @property
    @abstractmethod
    def mpp_x(self) -> float:
        """Resolution of the image in meters per pixel along x-axis"""
        raise NotImplementedError

    @property
    @abstractmethod
    def mpp_y(self) -> float:
        """Resolution of the image in meters per pixel along y-axis"""
        raise NotImplementedError

    @property
    @abstractmethod
    def mpp_z(self) -> float:
        """Resolution of the image in meters per pixel along z-axis"""
        raise NotImplementedError

    @property
    @abstractmethod
    def channel_names(self) -> list[str] | None:
        """Names of the microscopy channels"""
        raise NotImplementedError

    @property
    @abstractmethod
    def image_type(self) -> str:
        """Indicator of the original image format/microscopy vendor"""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_file(path: str):
        """Load metadata from microscopy image in path"""
        raise NotImplementedError


class CZIImageMetadata(ImageMetadata):
    metadata: dict[str, Any]

    @computed_field
    @property
    def image_type(self) -> str:
        return "czi"

    def _parse_channel_id(self, channel_name: str) -> int:
        """Parse CZI channel id representation to channel index"""
        if channel_name is None:
            return
        return int(channel_name.replace("Channel:", ""))

    @computed_field
    @property
    def _channel_info(self) -> list[dict[str, str]]:
        """Obtain channel metadata from CZI metadata file

        Notes
        -----
        CZI represents strings in the `Channel` metadata field as list of dicts.
        The dict minimally contains an `@ID` and a `PixelType` key, but
        may also contain a `Name` key.
        """
        channels = (
            self.metadata.get("ImageDocument", {})
            .get("Metadata", {})
            .get("Information", {})
            .get("Image", {})
            .get("Dimensions", {})
            .get("Channels", {})
            .get("Channel")
        )

        # For a single channel, a dict is returned
        if isinstance(channels, dict):
            channels = [channels]

        return channels or []

    @computed_field
    @property
    def channel_id(self) -> list[int]:
        """Parse channel metadata to list of channel ids

        Notes
        -----
        Per channel, IDs are stored under the key `@Id` in the form `Channel:<channel id>`
        in the channel metadata
        """
        return [self._parse_channel_id(channel.get("@Id")) for channel in self._channel_info]

    @computed_field
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

    @computed_field
    @property
    def _mpp(self) -> dict[str, dict[str, str]]:
        """Parse pixel resolution from slide image

        Note
        ----
        Pixel resolution is stored in `Distance` field and always specified in meters per pixel
        """
        mpp = (
            self.metadata.get("ImageDocument", {})
            .get("Metadata", {})
            .get("Scaling", {})
            .get("Items", {})
            .get("Distance", [])
        )

        # Transpose list of dictionaries to dictionary with dimension name (X, Y, Z)
        # as keys and data as values
        mpp = {dim.get("@Id", str(idx)): dim for idx, dim in enumerate(mpp)}

        return mpp

    @computed_field
    @property
    def mpp_x(self) -> float | None:
        """Return resolution in X dimension in [meters per pixel]"""
        mpp_x = self._mpp.get("X", {}).get("Value", None)
        return float(mpp_x) if mpp_x else None

    @computed_field
    @property
    def mpp_y(self) -> float | None:
        """Resolution in Y dimension in [meters per pixel]"""
        mpp_y = self._mpp.get("Y", {}).get("Value", None)
        return float(mpp_y) if mpp_y else None

    @computed_field
    @property
    def mpp_z(self) -> float | None:
        """Resolution in Z dimension in [meters per pixel]"""
        mpp_z = self._mpp.get("Z", {}).get("Value", None)
        return float(mpp_z) if mpp_z else None

    @computed_field
    @property
    def objective_name(self) -> str | None:
        """Utilized objective name. Required to infer objective_nominal_magnification

        Note
        ----
        Objective Name is stored as string in `ObjectiveName` field. Presumably,
        this represents the currently utilized objective
        """
        return (
            self.metadata.get("ImageDocument", {})
            .get("Metadata", {})
            .get("Scaling", {})
            .get("AutoScaling", {})
            .get("ObjectiveName")
        )

    @computed_field
    @property
    def objective_nominal_magnification(self) -> float | None:
        """Utilized objective_nominal_magnification

        Note
        ----
        Given the utilized objective the utilized objective_nominal_magnification can be extracted
        from the metadata on all available Objectives. The objective_nominal_magnification of an objective
        is given as `NominalMagnification` field.
        """
        objectives = (
            self.metadata.get("ImageDocument", {})
            .get("Metadata", {})
            .get("Information", {})
            .get("Instrument", {})
            .get("Objectives", {})
            .get("Objective", {})
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
