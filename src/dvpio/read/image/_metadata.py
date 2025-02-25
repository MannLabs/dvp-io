from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, computed_field


class ImageMetadata(BaseModel, ABC):
    metadata: dict[str, dict | list | str]

    @property
    @abstractmethod
    def magnification(self) -> int | None:
        raise NotImplementedError

    @property
    @abstractmethod
    def mpp_x(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def mpp_y(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def mpp_z(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def channel_names(self) -> list[str] | None:
        raise NotImplementedError

    @property
    @abstractmethod
    def image_type(self) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_file(path: str):
        raise NotImplementedError


class CZIImageMetadata(ImageMetadata):
    metadata: dict[str, Any]

    @computed_field
    @property
    def image_type(self) -> str:
        return "czi"

    def parse_channel_id(self, channel_name: str) -> int:
        """Parse CZI channel id representation to channel index"""
        if channel_name is None:
            return
        return int(channel_name.replace("Channel:", ""))

    @computed_field
    @property
    def channel_info(self) -> list[dict[str, str]]:
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
        return [self.parse_channel_id(channel.get("@Id")) for channel in self.channel_info]

    @computed_field
    @property
    def channel_names(self) -> list[str]:
        """Parse channel metadata to list of channel ids

        Notes
        -----
        Per channel, names are stored under the key `@Name` as str
        in the channel metadata
        """
        return [channel.get("@Name", str(idx)) for idx, channel in enumerate(self.channel_info)]

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
