For AI In Robotics

# Team
| Connor McGannon | Mattia Candotti | Ayberk Yetkin |
| :---: | :---: | :---: |
| ![Connor](media/Connor.jpg) | ![Mattia](media/Mattia.jpg) | ![Ayberk](media/Ayberk.jpg) |
| Simulation Design and Setup | Point Cloud Processing | Car Training |

# Full System
This system trains a car to travel from one side of a city map to the other by following a few key steps:

1. Segments real PointCloud LiDAR data from the NSW Spatial Services (see [Point Cloud Segmentation](#point-cloud-segmentation))
2. Uses that segmented data to construct a simulation environment representing all the buildings (see [Environment Building](#environment-building))
3. Trains a robot car to travel from one side of the city to a goal on the other (see [AI Car Training](#ai-car-training-and-testing))

## Installation, Setup, and Usage Guide

1. Create a virtual environment:
```bash
py -m venv .venv
```

2. Activate the virtual environment:
```bash
.\.venv\Scripts\Activate.ps1
```
!!! note "Usage Note"
    The .ps1 script is used for windows, if you are running on another OS, please consult python docs for activating virtual environments

3. Install the requirements list:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

!!! note "External Requirement"
    To install pybullet (one of the requirements), a C++ compiler must be installed. To use the standard microsoft MSVC compiler, click here: [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/). When the program launches, install the "Desktop development with C++" package and restart VS code before attempting installation again.

!!! note "Alternative Method"
    This repository comes with a setup script! Run ```.\setup.ps1``` in terminal to automatically execute the steps above!

# Point Cloud Segmentation

Point cloud segmentation is conducted through a 3 step process:

1. A .laz file is loaded using the load_laz function in clustering.py (see [load_laz](#api_load))
!!! note "Format Note"
    A valid pointcloud for this repository is expected to be formatted in the same way as NSW Spatial Data where a scalar label of 2 is considered the ground, and 6 is considered the buildings.
!!! note "Tip"
    This repository contains examples of readily useable pointcloud data here: [PointClouds](https://github.com/FrozenBreadstick/AIIR-TheCoolKindOfAI/tree/main/pointclouds). See folder 1 and 2 for raw data, or use the preprocessed files for spawning the environment.

2. The extracted point cloud data and ground truths are passed through a Random Forest Classifier (see [FelicityRandomForest](#api_forest)) to predict that is, and is not the ground vs the buildings. This is used to strip away the ground from the buildings for segmentation.

3. A DBSCAN algorithm (see [DavidBentleyScan](#api_dbscan)) is run on only the building points to cluster them into blocks.

After this, a small bit of logic is run in order to determine the bounds of each cluster such that it may be represented in the simulation environment for the car. This is done in a simple way by simply identifying the furthest point in all 4 directions of each cluster forming a 4 sided polygon for simplicity. (see [CedricCentroid](#api_centroid)) 

## Usage Guide


# Environment Building

Blah blah


# AI Car Training and Testing

blah blah




# API Reference
## clustering.py
### main {: #api_cluster_main }
::: clustering.main
### load_laz {: #api_load }
::: clustering.load_laz
### visualize {: #api_visualise }
::: clustering.visualize
### DavidBentleyScan {: #api_dbscan }
::: clustering.DavidBentleyScan
### FelicityRandomForest {: #api_forest }
::: clustering.FelicityRandomForest
### CericCentroid {: #api_centroid }
::: clustering.CedricCentroid


# Examples for documentation:
## Heading 2
**Bold Text** and *Italic Text*

- Bulleted list item
- Another item

!!! note "Installation Note"
    Make sure you have Python 3.10+ installed before continuing.

# Regular image
![My Image](media/test.png)

# Resized image
![My Image](media/test.png){ width="300" }

# Animated GIF (works exactly the same way)
![Demo Animation](media/guitar.gif)

<video controls width="100%">
  <source src="media/both.mp4" type="video/mp4">
</video>
