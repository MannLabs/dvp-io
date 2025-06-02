from collections.abc import Mapping
from typing import Any

import anndata as ad
import pandas as pd
from alphabase.psm_reader.psm_reader import psm_reader_provider
from spatialdata.models import TableModel

from dvpio._utils import experimental_docs, experimental_log

from ._anndata import AnnDataFactory


def available_reader() -> list[str]:
    """Get a list of all available readers, as provided by alphabase"""
    return sorted(psm_reader_provider.reader_dict.keys())


def _parse_pandas_index(index: pd.Index | pd.MultiIndex, set_index: str | None = None) -> pd.DataFrame:
    """Parse pandas index to pandas dataframe with object index

    Parameters
    ----------
    index
        :class:`pandas.Index`, will be parsed to :class:`pandas.DataFrame`
    set_index
        Defaults to None. Whether to set a column in the dataframe as the new index. If None,
        returns dataframe with range of type string as index

    Returns
    -------
    pd.DataFrame
        DataFrame with index values as columns, optionally with the column specified in `set_index`
        as index.
    """
    df = index.to_frame(index=False)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)

    if set_index is not None:
        df.set_index(set_index, inplace=True)

    return df


def parse_df(
    df: pd.DataFrame, obs_index: str | None = None, var_index: str | None = None, **table_kwargs
) -> ad.AnnData:
    """Convert a pandas dataframe to :class:`anndata.AnnData`

    Parameters
    ----------
    df
        Pandas dataframe of shape N (samples) x F (features). Expects observations (e.g. cells, samples) in rows
        and features (protein groups) in columns
    obs_index
        Name of dataframe column that should be set to index in `.obs` attribute
        (anndata.AnnData.var_names)
    var_index
        Name of dataframe column that should be set to index in `.obs` attribute
        (anndata.AnnData.var_names)
    **table_kwargs
        Keyword arguments passed to :meth:`spatialdata.models.TableModel.parse`

    Returns
    -------
    :class:`anndata.AnnData`
        AnnData object with N observations and F features.

            - .obs Contains content of df.index
            - .var contains content of df.columns

    Example
    -------
    .. code-block:: python

        import numpy as np
        import pandas as pd
        from dvpio.read.omics import parse_df

        df = pd.DataFrame(np.arange(9).reshape(3, 3), columns=["G1", "G2", "G3"], index=["A", "B", "C"])
        df = df.rename_axis(columns="gene", index="sample")

        adata = parse_df(df)

        assert adata.shape == (3, 3)
        assert "sample" in adata.obs.columns
        assert "gene" in adata.var.columns

        adata = parse_df(df, obs_index="sample")
        assert "sample" not in adata.obs.columns
        assert adata.obs.index.name == "sample"
    """
    X = df.to_numpy()

    obs = _parse_pandas_index(df.index, set_index=obs_index)
    var = _parse_pandas_index(df.columns, set_index=var_index)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    return TableModel.parse(adata, **table_kwargs)


@experimental_log
@experimental_docs
def read_precursor_table(
    path: str,
    reader_type: str,
    *,
    intensity_column: str | None = None,
    protein_id_column: str | None = None,
    raw_name_column: str | None = None,
    reader_kwargs: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> ad.AnnData:
    """Parse proteomics precursor reports to the :class:`anndata.AnnData` format

    Supported formats include

        - AlphaDIA `alphadia_parquet` (.parquet) `alphadia_tsv` (.tsv)
        - DIANN `diann` (.tsv)
        - MaxQuant
        - MSFragger `msfragger`
        - Sage `sage_parquet` (.parquet), `sage_tsv` (.tsv)
        - Spectronaut

    see :func:`dvpio.read.omics.available_reader` for a complete list

    Parameters
    ----------
    path
        Path to proteomics report
    reader_type
        Name of engine output, pass the method name of the corresponding reader. You can
        list all available readers with the :func:`dvpio.read.omics.available_reader` helper function
    intensity_column
        Column name of precursor intensity in report
    protein_id_column
        Column name of feature (i.e. protein group) in report
    raw_name_column
        Column names of individual samples in report.
    reader_kwargs
        Optional keyword arguments passed to :class:`alphabase.psm_reader.psm_reader.PSMReaderBase`
    kwargs
        Passed to :meth:`spatialdata.models.TableModel.parse`

    Returns
    -------
    :class:`anndata.AnnData`
        AnnData object that can be further processed with scVerse packages.

        - adata.X
            Stores values of the `intensity_column` argument the report as sparse matrix of shape observations x features
        - adata.obs
            Stores observations
        - adata.var
            Stores features

    Example
    -------

    .. code-block:: python

        from dvpio.io.read.omics import read_report, available_reader

        print(available_reader())
        > ['alphadia', 'alphadia_parquet', 'alphapept', 'diann', 'maxquant', ...]

        path = ...
        adata = read_precursor_table(
            path,
            reader_type="diann",
            intensity_column="Precursor.Normalised",
            raw_name_column="File.Name",
            protein_id_column="Protein.Names"
        )

    """
    if reader_type not in available_reader():
        raise ValueError(f"Argument reader_type must be one of {''.join(available_reader())}, not {reader_type}")

    reader_kwargs = {} if reader_kwargs is None else reader_kwargs

    factory = AnnDataFactory.from_files(
        path,
        reader_type=reader_type,
        intensity_column=intensity_column,
        protein_id_column=protein_id_column,
        raw_name_column=raw_name_column,
        **reader_kwargs,
    )

    adata = factory.create_anndata()

    return TableModel.parse(adata, **kwargs)
