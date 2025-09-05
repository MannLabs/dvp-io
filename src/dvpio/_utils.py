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


def deprecated_docs(func):
    """Decorator to mark a function as deprecated in the docstring."""
    func.__doc__ = f"""**Warning: This function is deprecated and will be removed in the next minor release**\n\n
    {func.__doc__ or ""}"""
    return func


def deprecated_log(message=None):
    """Decorator to mark a function as deprecated with a warning log.

    Parameters
    ----------
    message
        Optional custom deprecation message. If not provided, uses default format.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if message is None:
                warning_message = f"Function {func.__name__} is deprecated and will be removed in future versions."
            else:
                warning_message = message

            warnings.warn(
                warning_message,
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    # Handle both @deprecated_log and @deprecated_log() usage
    if callable(message):
        func = message
        message = None
        return decorator(func)

    return decorator
