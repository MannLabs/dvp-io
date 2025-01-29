import anndata as ad
import pandas as pd
from alphabase.psm_reader.psm_reader import psm_reader_provider
from spatialdata.models import TableModel


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

        adata = parse_df(df, obs_index="sample)
        assert "sample" not in adata.obs.columns
        assert adata.obs.index.name == "sample"
    """
    X = df.to_numpy()

    obs = _parse_pandas_index(df.index, set_index=obs_index)
    var = _parse_pandas_index(df.columns, set_index=var_index)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    return TableModel.parse(adata, **table_kwargs)
