# dvp-io

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]
[![codecov](https://codecov.io/gh/lucas-diedrich/dvp-io/graph/badge.svg?token=RO2UBP3JQ0)](https://codecov.io/gh/lucas-diedrich/dvp-io)

[badge-tests]: https://img.shields.io/github/actions/workflow/status/lucas-diedrich/dvp-io/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/dvp-io

Read and write funtionalities from and to spatialdata for deep visual proteomics

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][],
[tutorials][],
and the [FAQs][].

## Installation

You need to have Python 3.10 or newer installed on your system.

<!--
1) Install the latest release of `dvp-io` from [PyPI][]:

```bash
pip install dvp-io
```
-->

### Users

Install the latest development version:

```bash
# Optional: Create a suitable conda envionemnt
conda create -n dvpio python=3.11 -y  && conda activate dvpio
```

```bash
pip install git+https://github.com/lucas-diedrich/dvp-io.git@main
```

#### C++ dependencies

Some critical dependencies of `dvpio` require C++ bindings, so a suitable C++ compiler must be installed.

##### For Unix Users (Linux, macOS)

Ensure `cmake` and `libssh2` are installed by running:

```shell
# Unix
conda install -n dvpio conda-forge::cmake conda-forge::libssh2
```

##### Windows users

Windows users require the **Microsoft Visual C++ (MSVC) compiler**. Before creating the dvpio environment, follow these steps:

1. Download and install [Visual Studio](https://visualstudio.microsoft.com/downloads/).
2. In the installer, select **Desktop Development with C++** as a workload.
3. Complete the installation and restart your system if necessary.

After installation, proceed with the dvp-io installation steps above.

## Release notes

Refer to the [Releases page](https://github.com/lucas-diedrich/dvp-io/releases) for information on releases and the changelog.

## References

> SPARCS, a platform for genome-scale CRISPR screening for spatial cellular phenotypes Niklas Arndt Schmacke, Sophia Clara Maedler, Georg Wallmann, Andreas Metousis, Marleen Berouti, Hartmann Harz, Heinrich Leonhardt, Matthias Mann, Veit Hornung bioRxiv 2023.06.01.542416; doi: https://doi.org/10.1101/2023.06.01.542416

> Marconato, L. et al. SpatialData: an open and universal data framework for spatial omics. Nat Methods 1–5 (2024) doi:10.1038/s41592-024-02212-x.

> Zeng, W.-F. et al. AlphaPeptDeep: a modular deep learning framework to predict peptide properties for proteomics. Nat Commun 13, 7238 (2022).

[mambaforge]: https://github.com/conda-forge/miniforge#mambaforge
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/lucas-diedrich/dvp-io/issues
[tests]: https://github.com/lucas-diedrich/dvp-io/actions/workflows/test.yml
[documentation]: https://dvp-io.readthedocs.io
[changelog]: https://dvp-io.readthedocs.io/en/latest/changelog.html
[api documentation]: https://dvp-io.readthedocs.io/en/latest/api.html
[FAQs]: https://dvp-io.readthedocs.io/en/latest/faq.html
[tutorials]: https://dvp-io.readthedocs.io/en/latest/tutorials.html
[pypi]: https://pypi.org/project/dvp-io
