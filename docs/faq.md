# FAQ

## Current scope of the package

Please raise an [issue](https://github.com/lucas-diedrich/dvp-io/issues) to request support for additonal data formats.

### Images (Tested)

| Type  |                                        | Function         | Supported channels       | Wrapped library                    |
| ----- | -------------------------------------- | ---------------- | ------------------------ | ---------------------------------- |
| .czi  | Fluorescence Microscopy Single-Channel | `read_czi`       | Grayscale                | pylibczirw                         |
| .czi  | Fluorescence Microscopy Multi-channel  | `read_czi`       | Grayscale                | pylibczirw                         |
| .czi  | Whole Slide Image                      | `read_czi`       | RGB(A)                   | pylibczirw                         |
| .mrxs | Whole Slide Images                     | `read_openslide` | RGB(A)                   | openslide                          |
| .tiff | -                                      | `read_custom`    | (multichannel) grayscale | dask.array.image/skimage.io.imread |

### Images (supported, in principle)

| Type  |                             | Function         | Supported channels | Wrapped library                    |
| ----- | --------------------------- | ---------------- | ------------------ | ---------------------------------- |
| .svs  | Whole Slide Images (Aperio) | `read_openslide` | RGB(A)             | openslide                          |
| .dcm  | Whole Slide Images (DICOM)  | `read_openslide` | RGB(A)             | openslide                          |
| .ndpi | Hamamatsu                   | `read_openslide` | RGB(A)             | openslide                          |
| .svs  | Whole Slide Images          | `read_openslide` | RGB(A)             | openslide                          |
| .jpeg | -                           | `read_custom`    | RGB(A), grayscale  | dask.array.image/skimage.io.imread |
| .png  | -                           | `read_custom`    | RGB(A), grayscale  | dask.array.image/skimage.io.imread |
| .tiff | -                           | `read_custom`    | RGB(A)             | dask.array.image/skimage.io.imread |

### Shapes

| Type |     | Function                     | Wrapped library |
| ---- | --- | ---------------------------- | --------------- |
| .xml | LMD | `dvpio.read.shapes.read_lmd` | py-lmd          |

### Omics

| Type               |                                            | Function                                               | Wrapped library |
| ------------------ | ------------------------------------------ | ------------------------------------------------------ | --------------- |
| `pandas.DataFrame` | Any type, preprocessed into correct format | `dvpio.read.omics.parse_df`                            | -               |
| .tsv               | alphaDIA                                   | `dvpio.read.shapes.read_precursor_table` (alphadia)    | alphabase       |
| .tsv               | DIANN                                      | `dvpio.read.shapes.read_precursor_table` (diann)       | alphabase       |
| .tsv               | DIANN                                      | `dvpio.read.shapes.read_precursor_table` (diann)       | alphabase       |
| .tsv               | alphapept                                  | `dvpio.read.shapes.read_precursor_table` (alphapept)   | alphabase       |
| .tsv               | MSFragger                                  | `dvpio.read.shapes.read_precursor_table` (msfragger)   | alphabase       |
| .tsv               | DIANN                                      | `dvpio.read.shapes.read_precursor_table` (msfragger)   | alphabase       |
| .tsv               | spectronaut                                | `dvpio.read.shapes.read_precursor_table` (spectronaut) | alphabase       |
| .parquet           | alphaDIA                                   | `dvpio.read.shapes.read_precursor_table` (alphadia)    | alphabase       |
| .parquet           | DIANN                                      | `dvpio.read.shapes.read_precursor_table` (diann)       | alphabase       |

## How to...

### ... open spatialdata in Napari?

This requires you to have `napari_spatialdata` installed in the respective environment
In a **jupyter notebook**, you can use the following snippet:

```python
import spatialdata
from napari_spatialdata import Interactive

sdata = spatialdata.read_zarr("/path/to/sdata.zarr")
session = Interactive(sdata)
session.run()
```

You can also **import it directly from the napari viewer**.
Open the napari viewer, e.g. from the commandline.

```bash
> conda activate <my_env>
> napari
```

In napari, go to `File > Open Directory` (or use the shortcut `Cmd+Shift+O`) and go to the storage location of your spatialdata object. Select the `napari spatialdata` reader in the pop up menu.

## Known Issues

Please raise an [issue](https://github.com/lucas-diedrich/dvp-io/issues) to report bugs.

### Import of `napari-spatialdata` fails

Importing `napari_spatialdata` might initially fail due to missing non-python dependencies. If you get the following error:

```python
import napari_spatialdata
> qtpy.PythonQtError: No Qt bindings could be found
```

Try to install the `pyqt5-tools` binaries in your environment

```bash
pip install pyqt5-tools
```

### Rendering cell segmentation results takes very long in Napari

This is a known issue in Napari. Very recently, Grzegorz Bokota and colleagues implemented an experimental C++ version of the rendering algorithm for shapes that greatly improves the performance (~10x). See their [blogpost](https://napari.org/island-dispatch/blog/triangles_speedup_beta.html). To use the feature install napari with optional dependencies:

```shell
pip install "napari[optional,pyqt6]>=0.5.6rc0"
```

And tick the box in the Napari GUI under `Napari > Preferences > Experimental > Use C++ code to speed up creation and updates of Shapes Layers`

### I can't overlay multiple channels for my fluorescence image in the Napari viewer

On the left, select the image layer you are interested in, right click, and select `Split Stack`. Now, the individual channels are shown as distinct layers.
