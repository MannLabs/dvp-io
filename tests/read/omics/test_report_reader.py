import numpy as np
import pandas as pd
import pytest
from alphabase.psm_reader.psm_reader import psm_reader_provider
from spatialdata.models import TableModel

from dvpio.read.omics import available_reader, parse_df
from dvpio.read.omics.report_reader import _parse_pandas_index

# Test data
gene_index = pd.Index(["G1", "G2", "G3"], name="gene")
sample_index = pd.Index(["A", "B", "C"], name="sample")
index_complex = pd.MultiIndex.from_arrays([np.arange(3), np.arange(3, 6)], names=["A", "B"])
df = pd.DataFrame(np.arange(9).reshape(3, 3), columns=gene_index, index=sample_index)
df_complex = pd.DataFrame(np.arange(9).reshape(3, 3), columns=gene_index, index=index_complex)


def test_available_reader() -> None:
    list_of_available_reader = available_reader()

    assert len(list_of_available_reader) == len(psm_reader_provider.reader_dict)
    assert "alphadia" in list_of_available_reader


@pytest.mark.parametrize(
    ("index", "set_index", "shape", "columns"),
    [
        (gene_index, None, (3, 1), ["gene"]),
        (gene_index, "gene", (3, 0), None),
        (sample_index, None, (3, 1), ["sample"]),
        (sample_index, "sample", (3, 0), None),
        (index_complex, None, (3, 2), ["A", "B"]),
        (index_complex, "A", (3, 1), ["B"]),
    ],
)
def test_parse_pandas_index(
    index: pd.Index, set_index: None | str, shape: tuple[int], columns: list[str] | None
) -> None:
    df = _parse_pandas_index(index, set_index=set_index)

    assert df.shape == shape
    if columns is not None:
        assert all(df.columns == columns)


@pytest.mark.parametrize(
    ["df", "obs_index", "var_index", "obs_shape", "var_shape"],
    [
        (df, None, None, 1, 1),
        (df, "sample", None, 0, 1),
        (df, None, "gene", 1, 0),
        (df, "sample", "gene", 0, 0),
        (df_complex, None, None, 2, 1),
        (df_complex, "A", None, 1, 1),
    ],
)
def test_parse_df(
    df: pd.DataFrame, obs_index: str | None, var_index: str | None, obs_shape: tuple[int], var_shape: tuple[int]
) -> None:
    df = df.copy(deep=True)
    adata = parse_df(df, obs_index=obs_index, var_index=var_index)

    TableModel().validate(adata)

    assert adata.shape == df.shape
    assert adata.obs.shape[1] == obs_shape
    assert adata.var.shape[1] == var_shape
    assert adata.obs.index.name == obs_index
    assert adata.var.index.name == var_index


@pytest.mark.parametrize(
    ["df", "obs_index", "var_index", "obs_shape", "var_shape"],
    [
        (df, None, None, 1, 1),
        (df, None, "gene", 1, 0),
    ],
)
def test_parse_df_table_kwargs(
    df: pd.DataFrame, obs_index: str | None, var_index: str | None, obs_shape: int, var_shape: int
) -> None:
    """Test whether matching shapes attributes works"""
    df = df.copy(deep=True)
    df["region_key"] = "region1"
    df.set_index("region_key", append=True, inplace=True)

    adata = parse_df(
        df, obs_index=obs_index, var_index=var_index, instance_key="sample", region_key="region_key", region="region1"
    )

    TableModel().validate(adata)
    assert adata.shape == df.shape


def test_read_precursor_table():
    pass
