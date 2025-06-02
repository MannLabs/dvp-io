import numpy as np
import pandas as pd
import pytest
from alphabase.psm_reader.psm_reader import psm_reader_provider
from spatialdata.models import TableModel

from dvpio.read.omics import available_reader, parse_df, read_precursor_table
from dvpio.read.omics.report_reader import _parse_pandas_index


@pytest.fixture()
def gene_index():
    """Feature-like pandas index"""
    return pd.Index(["G1", "G2", "G3"], name="gene")


@pytest.fixture()
def sample_index() -> pd.Index:
    """Sample-like pandas index"""
    return pd.Index(["A", "B", "C"], name="sample")


@pytest.fixture()
def sample_index_int() -> pd.Index:
    """Sample-like pandas index with integer as column name"""
    return pd.Index(["A", "B", "C"], name=0)


@pytest.fixture()
def multi_index() -> pd.MultiIndex:
    """Pandas multiindex"""
    return pd.MultiIndex.from_arrays([np.arange(3), np.arange(3, 6)], names=["A", "B"])


@pytest.fixture()
def df(gene_index: pd.Index, sample_index: pd.Index) -> pd.DataFrame:
    """Pandas dataframe with single-level indices"""
    return pd.DataFrame(np.arange(9).reshape(3, 3), columns=gene_index, index=sample_index)


@pytest.fixture()
def df_int(gene_index: pd.Index, sample_index_int: pd.Index) -> pd.DataFrame:
    """Pandas dataframe with index with int column name"""
    return pd.DataFrame(np.arange(9).reshape(3, 3), columns=gene_index, index=sample_index_int)


@pytest.fixture()
def df_complex(gene_index: pd.Index, multi_index: pd.MultiIndex) -> pd.DataFrame:
    """Pandas dataframe with multiindex"""
    return pd.DataFrame(np.arange(9).reshape(3, 3), columns=gene_index, index=multi_index)


@pytest.fixture()
def alphadia_pg_report() -> pd.DataFrame:
    """Read alphaDIA protein group report and return it as N x F matrix"""
    df = pd.read_csv("./data/omics/alphadia/alphadia.protein-group.tsv", sep="\t", index_col="pg")
    return df.T


def test_available_reader() -> None:
    list_of_available_reader = available_reader()

    assert len(list_of_available_reader) == len(psm_reader_provider.reader_dict)
    assert "alphadia" in list_of_available_reader


@pytest.mark.parametrize(
    ("set_index", "shape", "columns"),
    [
        (None, (3, 1), ["gene"]),
        ("gene", (3, 0), None),
    ],
)
def test_parse_pandas_index(gene_index, set_index: None | str, shape: tuple[int], columns: list[str] | None) -> None:
    df = _parse_pandas_index(gene_index, set_index=set_index)

    assert df.shape == shape
    if columns is not None:
        assert all(df.columns == columns)


@pytest.mark.parametrize(("set_index", "shape", "columns"), [(None, (3, 1), "0")])
def test_parse_pandas_index_int(
    sample_index_int, set_index: None | str, shape: tuple[int], columns: list[str] | None
) -> None:
    df = _parse_pandas_index(sample_index_int, set_index=set_index)

    assert df.shape == shape
    if columns is not None:
        assert all(df.columns == columns)


@pytest.mark.parametrize(("set_index", "shape", "columns"), [(None, (3, 2), ["A", "B"]), ("A", (3, 1), ["B"])])
def test_parse_pandas_multi_index(multi_index, set_index: str | None, shape: tuple[int], columns: str | None) -> None:
    df = _parse_pandas_index(multi_index, set_index=set_index)

    assert df.shape == shape
    if columns is not None:
        assert all(df.columns == columns)


@pytest.mark.parametrize(
    ["obs_index", "var_index", "obs_shape", "var_shape"],
    [
        (None, None, 1, 1),
        ("sample", None, 0, 1),
        (None, "gene", 1, 0),
        ("sample", "gene", 0, 0),
    ],
)
def test_parse_df(
    df, obs_index: str | None, var_index: str | None, obs_shape: tuple[int], var_shape: tuple[int]
) -> None:
    df = df.copy(deep=True)
    adata = parse_df(df, obs_index=obs_index, var_index=var_index)

    TableModel().validate(adata)

    assert adata.shape == df.shape
    assert adata.obs.shape[1] == obs_shape
    assert adata.var.shape[1] == var_shape
    assert adata.obs.index.name == obs_index
    assert adata.var.index.name == var_index


@pytest.mark.parametrize(["obs_index", "var_index", "obs_shape", "var_shape"], [(None, "gene", 1, 0)])
def test_parse_df_int_index(
    df_int, obs_index: str | None, var_index: str | None, obs_shape: tuple[int], var_shape: tuple[int]
) -> None:
    df = df_int.copy(deep=True)
    adata = parse_df(df, obs_index=obs_index, var_index=var_index)

    TableModel().validate(adata)

    assert adata.shape == df.shape
    assert adata.obs.shape[1] == obs_shape
    assert adata.var.shape[1] == var_shape
    assert adata.obs.index.name == obs_index
    assert adata.var.index.name == var_index


@pytest.mark.parametrize(["obs_index", "var_index", "obs_shape", "var_shape"], [(None, None, 2, 1), ("A", None, 1, 1)])
def test_parse_df_multi_index(
    df_complex, obs_index: str | None, var_index: str | None, obs_shape: tuple[int], var_shape: tuple[int]
) -> None:
    df = df_complex.copy(deep=True)
    adata = parse_df(df, obs_index=obs_index, var_index=var_index)

    TableModel().validate(adata)

    assert adata.shape == df.shape
    assert adata.obs.shape[1] == obs_shape
    assert adata.var.shape[1] == var_shape
    assert adata.obs.index.name == obs_index
    assert adata.var.index.name == var_index


@pytest.mark.parametrize(
    ["obs_index", "var_index", "obs_shape", "var_shape"],
    [
        (None, None, 1, 1),
        (None, "gene", 1, 0),
    ],
)
def test_parse_df_table_kwargs(
    df, obs_index: str | None, var_index: str | None, obs_shape: int, var_shape: int
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


@pytest.mark.parametrize(
    ["shape", "obs_index", "var_index", "obs_columns", "var_columns"],
    [((3, 7497), None, None, ["0"], ["pg"]), ((3, 7497), "0", "pg", None, None)],
)
def test_parse_df_real_data(
    alphadia_pg_report,
    shape: tuple[int],
    obs_index: str | None,
    var_index: str | None,
    obs_columns: list[str],
    var_columns: list[str],
) -> None:
    df = alphadia_pg_report
    adata = parse_df(df, obs_index=obs_index, var_index=var_index)

    assert adata.shape == shape

    if obs_columns is not None:
        assert adata.obs.columns == obs_columns
    if var_columns is not None:
        assert adata.var.columns == var_columns


@pytest.mark.parametrize(
    ("path", "reader_type", "func_kwargs", "shape"),
    [
        # Read protein groups
        ("./data/omics/alphadia/alphadia.precursors.tsv", "alphadia", {}, (3, 7497)),
        (
            "./data/omics/diann/diann_report.tsv",
            "diann",
            {"intensity_column": "PG.Normalised", "protein_id_column": "Protein.Group", "raw_name_column": "File.Name"},
            (3, 380),
        ),
        # Get precursor intensities
        (
            "./data/omics/diann/diann_report.tsv",
            "diann",
            {
                "intensity_column": "Precursor.Normalised",
                "protein_id_column": "Precursor.Id",
                "raw_name_column": "File.Name",
            },
            (3, 1088),
        ),
    ],
)
def test_read_precursor_table(path: str, reader_type: str, func_kwargs: dict, shape: tuple[int]) -> None:
    adata = read_precursor_table(path, reader_type=reader_type, **func_kwargs)
    assert adata.shape == shape
