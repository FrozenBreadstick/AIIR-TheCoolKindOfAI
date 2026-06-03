For AI In Robotics

# Team
| Connor McGannon | Mattia Candotti | Ayberk Yetkin |
| :---: | :---: | :---: |
| ![Connor](media/Connor.jpg) | ![Mattia](media/Mattia.jpg) | ![Ayberk](media/Ayberk.jpg) |
| Simulation Design and Setup | Point Cloud Processing | Car Training |

<hr style="border: 5px solid #3f51b5;">

# Full System
This system trains a car to travel from one side of a city map to the other by following a few key steps:

1. Segments real PointCloud LiDAR data from the NSW Spatial Services (see [Point Cloud Segmentation](clustering.md#point-cloud-segmentation))
2. Uses that segmented data to construct a simulation environment representing all the buildings (see [Environment Building](environment.md#environment-building))
3. Trains a robot car to travel from one side of the city to a goal on the other (see [AI Car Training](car.md#ai-car-training-and-testing))

## Results of Example Trained Model

<div style="display: flex; gap: 16px; flex-wrap: wrap; margin-top: 24px;">
  <div style="flex: 1; min-width: 280px;">
    <h3 style="margin-bottom: 8px;">Car almost clears corner</h3>
    <video controls width="100%">
      <source src="media\car_almost_gets_around_corner.mp4" type="video/mp4">
    </video>
    <p style="margin-top: 8px; font-size: 0.95em;">
      In this Video Agent can be seen clearly attempting to drive around the wall to the checkpoint. Unfortunetaly the checkpoint was only just too far away as the agent did eventually hit the wall.
    </p>
  </div>
  <div style="flex: 1; min-width: 280px;">
    <h3 style="margin-bottom: 8px;">Car successfully clears corner</h3>
    <video controls width="100%">
      <source src="media\car_succeeds_around_cornor.mp4" type="video/mp4">
    </video>
    <p style="margin-top: 8px; font-size: 0.95em;">
      In this Video Agent can be seen successfully avoiding a collision with a building that sticks out between the checkpoints. The agent rather than drive straight between the checkpoints avoids the corner of the building and goes on to finish the scenario.
    </p>
  </div>
</div>
<p style="margin-top: 8px; font-size: 0.95em;">
</p>

For a full demonstration of the system from start to end see [Demo Video](#demo-video)

---

## Installation and Setup

!!! note "Python Version"
    Ensure python 3.11+ is installed

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

---

## Usage Guide

To run the full system, run the main.py script from terminal after activating the virtual environment in the root directory of the repository:
```bash
python .\src\main.py
```

See below for a list of parameters:

| **Parameter** | **Type** | **Description** | **Default** |
| :--- | :--- | :--- | :--- |
| `--run_cluster` | boolean | Whether to run the clustering script before training/testing. | `True` |
| `--run_train` | boolean | Whether to run the training process. | `False` |
| `--run_test` | boolean | Whether to run the testing process immediately after. | `True` |
| `--checkpoint_freq` | integer | Checkpoint frequency in terms of how many goals spawn in the environment. | `30` |
| `--model_path` | string | Relative path to the model to load, start training from, or test. | `"tensor_data\ppo39.30_2\ppo_driving_6700000_steps"` |
| `--laz` | string | The filepath of the PointCloud laz file to use. | `"pointclouds/1/Denoise_NoVeg_Subsampled.laz"` |
| `--total_timesteps` | integer | Total timesteps for the training process. | `2` |
| `--n_envs` | integer | Number of parallel environments to run. | `8` |
| `--n_steps` | integer | Number of steps to run per environment before updating. | `1024` |
| `--batch_size` | integer | Minibatch size for the optimization. | `512` |
| `--n_epochs` | integer | Number of epochs when optimizing the surrogate loss. | `4` |
| `--learning_rate` | float | The learning rate of the optimizer. | `0.0001` |
| `--entropy_coef` | float | Entropy coefficient for the loss calculation. | `0.15` |
| `--gae_lambda` | float | Factor for trade-off of bias vs variance for Generalized Advantage Estimator. | `0.95` |
| `--gamma` | float | Discount factor. | `0.995` |
| `--max_grad_norm` | float | The maximum value for the gradient clipping. | `0.3` |
| `--clip_range` | float | Clipping parameter for the value function. | `0.2` |

Example command with all arguments specified:
```bash
python .\src\main.py --run_cluster True --run_train False --run_test True --checkpoint_freq 30 --model_path "tensor_data/ppo39.30_2/ppo_driving_6700000_steps" --laz "pointclouds/1/Denoise_NoVeg_Subsampled.laz" --total_timesteps 2 --n_envs 8 --n_steps 1024 --batch_size 512 --n_epochs 4 --learning_rate 0.0001 --entropy_coef 0.15 --gae_lambda 0.95 --gamma 0.995 --max_grad_norm 0.3 --clip_range 0.2      
```

The arguments above are all optional, you can use as many or as little as you like. A simple command for running training with defaults, but using a custom laz file would be:

```bash
python .\src\main.py --laz "pointclouds/1/Denoise_NoVeg_Subsampled.laz"
```

Newly trained models will be saved in the "model/" folder.

To adjust the training values and generally configure the system in an easier way please modify the "GLOBAL CONFIGURATION PARAMETERS" from the "main.py" file.

---

## Demo Video

Below is a video demonstrating how to run the system and what can be expected when you do

<video controls width="100%">
  <source src="media\AIIR-Demo.mp4" type="video/mp4">
</video>

For more detailed code descriptions please see the [API Docs](api.md#api-reference) 