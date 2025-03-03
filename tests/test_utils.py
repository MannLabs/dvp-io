import pytest

from dvpio._utils import experimental_docs, experimental_log, is_parsed


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
