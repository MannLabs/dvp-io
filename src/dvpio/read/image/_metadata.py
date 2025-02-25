from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


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
