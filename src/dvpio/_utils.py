import functools
import warnings


def experimental_docs(func):
    """Decorator to mark a function as experimental in the docstring."""
    func.__doc__ = "**Warning:** This function is experimental and may change in future versions. \n\n" + (
        func.__doc__ or ""
    )
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
