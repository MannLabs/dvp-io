# Spatialdata and DVP

## What is spatialdata?

Spatialdata {cite:p}`Spatialdata2024` is a data framework for the joint and accessible storage of imaging data, annotations, and -omics data. This is extremely useful for spatial -omics experiments, that commonly generate imaging data and (partially) paired -omics measurements.

### Yet another dataformat?

The power of spatialdata comes from the fact that it was designed from the start as integrated format for -omics and imaging data. The spatialdata format implements FAIR principles following (future versions of) the OME-Zarr storage/disk format. It further provides many convenient functionalities, including the overlay of imaging data and -omics measurements as static plots or in dynamic viewers (Napari), interfaces with -omics analysis tools, and deep learning frameworks. Spatialdata is performant and can handle large imaging, and -omics data, as well as complex annotations.

Most notably, spatialdata makes it simple to keep track of cells or shapes of interest between modalities (imaging, annotation, -omics)

It should be expected that spatialdata will become the de-facto standard for the analysis of spatial -omics data in the following years, at least in the Python ecosystem.

### Spatialdata and the DVP workflow

Spatialdata implements key functionalities that are relevant in the DVP workflow.

A typical DVP workflow is outlined in the following. Note that spatialdata implements storage options for every step of the workflow and thus helps to integrate the different relevant modalities.

| \#  | Step                                  | Modality                 | Format                                 | Spatial Element                          |
| --- | ------------------------------------- | ------------------------ | -------------------------------------- | ---------------------------------------- |
| 1   | Immunofluorescence/Pathology staining | Imaging                  | `.czi`, `.mrxs`, `.tiff`               | `.images`                                |
| 2   | Cell segmentation                     | Annotation               | cellpose, ... (e.g. `.tiff`)           | `.shapes` vectors, `.labels` raster data |
| 3   | Selection of cells                    | Annotation/Featurization | scPortrait (`diverse`)                 | `.tables`                                |
| 4   | Excision of cells                     | -                        | pyLMD {cite:p}`Sparcs2023` (`.xml`)    | -                                        |
| 5   | MS measurement                        | omics                    | alphaDIA, alphabase, DIANN (`diverse`) | `.tables`                                |

### DVP IO

DVP-IO is supposed to be lightweight and enable the integration of the `spatialdata` paradigm with the current standard DVP workflow. This means that it will implement read and write functionalities to and from spatialdata and current experimental DVP formats (segmentation masks, microscopy images, omics data). DVP-IO **will not** become a designated analysis tool for DVP data.
