import pytest

from dvpio._utils import deprecated_docs, deprecated_log, experimental_docs, experimental_log, is_parsed


@pytest.fixture()
def function_factory():
    def sample_func():
        """Original docstring."""
        pass

    return sample_func


def test_is_parsed(function_factory):
    sample_func = is_parsed(function_factory)

    assert sample_func._is_parsed


def test_experimental_docs(function_factory):
    sample_func = experimental_docs(function_factory)

    assert "Warning: This function is experimental" in sample_func.__doc__


def test_experimental_log(function_factory):
    sample_func = experimental_log(function_factory)

    with pytest.warns(UserWarning, match="is experimental and may change"):
        sample_func()


def test_deprecated_docs(function_factory):
    sample_func = deprecated_docs(function_factory)

    assert "Warning: This function is deprecated" in sample_func.__doc__


def test_deprecated_log_default_message(function_factory):
    """Test deprecated_log with default message using @ syntax."""
    sample_func = deprecated_log()(function_factory)

    with pytest.warns(
        DeprecationWarning, match="Function sample_func is deprecated and will be removed in future versions"
    ):
        sample_func()


def test_deprecated_log_custom_message(function_factory):
    """Test deprecated_log with custom message."""
    custom_message = "This function is obsolete. Use new_function() instead."
    sample_func = deprecated_log(custom_message)(function_factory)

    with pytest.warns(DeprecationWarning, match="This function is obsolete. Use new_function\\(\\) instead."):
        sample_func()


def test_deprecated_log_preserves_function_metadata(function_factory):
    """Test that deprecated_log preserves function name and docstring."""
    sample_func = deprecated_log(function_factory)

    assert sample_func.__name__ == function_factory.__name__
    assert sample_func.__doc__ == function_factory.__doc__


def test_deprecated_log_preserves_return_value(function_factory):
    """Test that deprecated_log preserves function return value."""

    def func_with_return():
        return "test_value"

    decorated_func = deprecated_log(func_with_return)

    with pytest.warns(DeprecationWarning):
        result = decorated_func()

    assert result == "test_value"


def test_deprecated_log_preserves_arguments():
    """Test that deprecated_log passes through arguments correctly."""

    def func_with_args(a, b, c=None):
        return (a, b, c)

    decorated_func = deprecated_log(func_with_args)

    with pytest.warns(DeprecationWarning):
        result = decorated_func(1, 2, c=3)

    assert result == (1, 2, 3)


def test_deprecated_log_warning_category():
    """Test that deprecated_log uses correct warning category."""

    def sample_func():
        pass

    decorated_func = deprecated_log(sample_func)

    with pytest.warns(DeprecationWarning):
        decorated_func()
