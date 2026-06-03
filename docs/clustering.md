# Point Cloud Segmentation

Point cloud segmentation is conducted through a 3 step process:

1. A .laz file is loaded using the load_laz function in clustering.py (see [load_laz](api.md#api_load))
!!! note "Format Note"
    A valid pointcloud for this repository is expected to be formatted in the same way as NSW Spatial Data where a scalar label of 2 is considered the ground, and 6 is considered the buildings.
!!! note "Tip"
    This repository contains examples of readily useable pointcloud data here: [PointClouds](https://github.com/FrozenBreadstick/AIIR-TheCoolKindOfAI/tree/main/pointclouds). See folder 1 and 2 for raw data, or use the preprocessed files for spawning the environment. More Point Cloud data can be found using the [ELVIS](https://elevation.fsdf.org.au/) website.

2. The extracted point cloud data and ground truths are passed through a Random Forest Classifier (see [FelicityRandomForest](api.md#api_forest)) to predict that is, and is not the ground vs the buildings. This is used to strip away the ground from the buildings for segmentation.

3. A DBSCAN algorithm (see [DavidBentleyScan](api.md#api_dbscan)) is run on only the building points to cluster them into blocks.

After this, a small bit of logic is run in order to determine the bounds of each cluster such that it may be represented in the simulation environment for the car. This is done in a simple way by simply identifying the furthest point in all 4 directions of each cluster forming a 4 sided polygon for simplicity. (see [CedricCentroid](api.md#api_centroid))

The full process is run in a single function called "main" (see [main](api.md#api_cluster_main))

---

## Usage Guide

The clustering methodology can be run standalone using the following:
```bash
python .\src\clustering.py
```

See below for a list of parameters:

| **Parameter** | **Type** | **Description** | **Default** |
| :--- | :--- | :--- | :--- |
| path | string | The filepath of the PointCloud laz file to use. | ```"pointclouds/1/Denoise_NoVeg_Subsampled.laz"``` |

Example command:
```bash
python src\clustering.py pointclouds/1/Denoise_NoVeg_Subsampled.laz
```

!!! note "Note"
    This usage guide is fundamentally identical to the full system guide.