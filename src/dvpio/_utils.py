import functools
import warnings
from collections.abc import Callable
from typing import Any


def is_parsed(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator function that marks a function as parsed by adding the `_is_parsed` attribute"""
    # Properties cannot be directly modified, modify getter function instead
    func._is_parsed = "_is_parsed"
    return func


def experimental_docs(func):
    """Decorator to mark a function as experimental in the docstring."""
    func.__doc__ = f"""**Warning: This function is experimental and may change in future versions**\n\n
    {func.__doc__ or ""}"""
    return func


def experimental_log(func):
    """Decorator to mark a function as experimental with a warning log."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"Function {func.__name__} is experimental and may change in future versions.",
            category=UserWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return wrapper
