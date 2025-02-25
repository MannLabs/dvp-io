import pytest

from dvpio._utils import experimental_docs, experimental_log


def test_experimental_docs():
    @experimental_docs
    def sample_func():
        """Original docstring."""
        pass

    assert "Warning: This function is experimental" in sample_func.__doc__


def test_experimental_log():
    @experimental_log
    def sample_func():
        pass

    with pytest.warns(UserWarning, match="is experimental and may change"):
        sample_func()
