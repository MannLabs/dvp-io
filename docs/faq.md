# FAQ

## Issues

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

## How to...

### ... open spatialdata in Napari?

In a **jupyter notebook**, you can use the following snippet:

```python
import spatialdata
from napari_spatialdata import Interactive

sdata = spatialdata.read_zarr("/path/to/sdata.zarr")
session = Interactive(sdata)
session.run()
```

You can also **import it directly from the napari viewer**.
Open the napari viewer, e.g. from the commandline

```bash
> conda activate <my_env>
> napari
```

In napari, go to `File > Open Directory` (or use the shortcut `Cmd+Shift+O`) and go to the storage location of your spatialdata object. Select the `napari spatialdata` reader in the pop up menu.
