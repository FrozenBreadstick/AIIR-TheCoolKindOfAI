For AI In Robotics

# Team
| Connor McGannon | Mattia Candotti | Ayberk Yetkin |
| :---: | :---: | :---: |
| ![Connor](media/Connor.jpg) | ![Mattia](media/Mattia.jpg) | ![Ayberk](media/Ayberk.jpg) |
| Simulation Design and Setup | Point Cloud Processing | Car Training |

<hr style="border: 5px solid #3f51b5;">

# Full System
This system trains a car to travel from one side of a city map to the other by following a few key steps:

1. Segments real PointCloud LiDAR data from the NSW Spatial Services (see [Point Cloud Segmentation](#point-cloud-segmentation))
2. Uses that segmented data to construct a simulation environment representing all the buildings (see [Environment Building](#environment-building))
3. Trains a robot car to travel from one side of the city to a goal on the other (see [AI Car Training](#ai-car-training-and-testing))

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
| --path | string | The filepath of the PointCloud laz file to use. | ```"pointclouds/1/Denoise_NoVeg_Subsampled.laz"``` |


TODO ADD TO HERE FOR ALL ARGUMENTS THAT EXIST
Example command with all arguments specified:
```bash
python src\main.py pointclouds/1/Denoise_NoVeg_Subsampled.laz        
```

Newly trained models will be saved in the "model/" folder.

To adjust the training values and generally configure the system please modify the "GLOBAL CONFIGURATION PARAMETERS" from the "main.py" file
---

## Demo Video

Below is a video demonstrating how to run the system and what can be expected when you do

<video controls width="100%">
  <source src="media\AIIR-Demo.mp4" type="video/mp4">
</video>

<hr style="border: 5px solid #6e00a1ca;">

<div style="display: flex; gap: 16px; flex-wrap: wrap; margin-top: 24px;">
  <div style="flex: 1; min-width: 280px;">
    <h3 style="margin-bottom: 8px;">Car almost clears corner</h3>
    <video controls width="100%">
      <source src="media\car_almost_gets_around_corner.mp4" type="video/mp4">
    </video>
  </div>
  <div style="flex: 1; min-width: 280px;">
    <h3 style="margin-bottom: 8px;">Car successfully clears corner</h3>
    <video controls width="100%">
      <source src="docs\media\car_succeeds_around_cornor.mp4" type="video/mp4">
    </video>
  </div>
</div>

# Point Cloud Segmentation

Point cloud segmentation is conducted through a 3 step process:

1. A .laz file is loaded using the load_laz function in clustering.py (see [load_laz](#api_load))
!!! note "Format Note"
    A valid pointcloud for this repository is expected to be formatted in the same way as NSW Spatial Data where a scalar label of 2 is considered the ground, and 6 is considered the buildings.
!!! note "Tip"
    This repository contains examples of readily useable pointcloud data here: [PointClouds](https://github.com/FrozenBreadstick/AIIR-TheCoolKindOfAI/tree/main/pointclouds). See folder 1 and 2 for raw data, or use the preprocessed files for spawning the environment. More Point Cloud data can be found using the [ELVIS](https://elevation.fsdf.org.au/) website.

2. The extracted point cloud data and ground truths are passed through a Random Forest Classifier (see [FelicityRandomForest](#api_forest)) to predict that is, and is not the ground vs the buildings. This is used to strip away the ground from the buildings for segmentation.

3. A DBSCAN algorithm (see [DavidBentleyScan](#api_dbscan)) is run on only the building points to cluster them into blocks.

After this, a small bit of logic is run in order to determine the bounds of each cluster such that it may be represented in the simulation environment for the car. This is done in a simple way by simply identifying the furthest point in all 4 directions of each cluster forming a 4 sided polygon for simplicity. (see [CedricCentroid](#api_centroid))

---

## Usage Guide

The clustering methodology can be run standalone using the following:
```bash
python .\src\clustering.py
```

See below for a list of parameters:

| **Parameter** | **Type** | **Description** | **Default** |
| :--- | :--- | :--- | :--- |
| --path | string | The filepath of the PointCloud laz file to use. | ```"pointclouds/1/Denoise_NoVeg_Subsampled.laz"``` |

Example command:
```bash
python .\src\clustering.py --path "pointclouds/1/Denoise_NoVeg_Subsampled.laz"
```

<hr style="border: 5px solid #c20000d7;">

# Environment Building

The simulation environment is implemented as a custom [Gymnasium](https://gymnasium.farama.org/) environment (`SimpleDrivingEnv`) built on top of [PyBullet](https://pybullet.org/). It takes the segmented point cloud data produced by the clustering step and constructs a physics-simulated city map that the car can drive through.
 
## How the Environment is Built
 
Each time the environment is reset, the following sequence runs:
 
**1. Load point cloud map data**
 
The `.npz` file produced by the clustering step is loaded. It contains three arrays:
 
| Array key | Contents |
| :--- | :--- |
| `metrics` | The 4-vertex boundary polygon of each building cluster |
| `centroids` | The (x, y) centroid of each building cluster |
| `min` / `max` | The bounding corners of the full map |
 
**2. Select a submap region**
 
A random (x, y) offset is sampled from the full map bounds. A 250 × 250 unit submap window is cut from this offset. Only buildings whose centroid falls within this window are spawned, so every episode presents a different section of the city.
 
**3. Spawn buildings as physics meshes**
 
Each building footprint is a convex quadrilateral from the clustering step. For each building in the submap:
 
- The 4 ground-plane vertices are extruded to height 1.0 to create 8 3D vertices forming a wall
- Triangle indices are generated manually for the 8 side faces (no top or bottom cap needed)
- A static PyBullet `GEOM_MESH` collision shape and red visual shape are created and added to the world
Buildings that would overlap the car spawn point or the goal are automatically skipped to guarantee a feasible episode.
 
**4. Update the A\* pathfinding grid**
 
As each building is spawned, its footprint (plus a 3-cell inflation buffer) is marked as unwalkable in a 250 × 250 integer grid. This grid is used in the next step for path planning.
 
**5. Spawn the car and goal**
 
- The **car** always spawns at the left-centre of the submap
- The **goal** always spawns at the right-centre of the submap, creating a consistent left-to-right traversal task
**6. Run A\* to find a path and place checkpoints**
 
An A\* search is run on the inflated occupancy grid from the car's start position to the goal. Every `checkpoint_frequency`-th node along the found path has a checkpoint goal object spawned at it. These checkpoints are consumed in order as the car passes through them, providing dense intermediate guidance rewards along a collision-free route.
 
**7. Spawn boundary walls**
 
Invisible mesh walls are placed around the submap perimeter (with a small buffer around the goal end zones) to prevent the car from driving off the edge of the map.
 
## Action and Observation Spaces
 
**Action space** — Continuous by default (as used in training):
 
| Dimension | Range | Description |
| :--- | :--- | :--- |
| Throttle | `[-1.0, 1.0]` | Forward/reverse drive |
| Steering | `[-0.6, 0.6]` | Left/right steering angle (radians) |
 
A discrete mode (`isDiscrete=True`) is also available, mapping 9 actions to the Cartesian product of `{-1, 0, 1}` throttle × `{-0.6, 0, 0.6}` steering.
 
**Observation space** — A flat vector of 40 floats:
 
| Elements | Shape | Description |
| :--- | :--- | :--- |
| Relative goal position | `(2,)` | (x, y) of the main goal in the car's local frame, range `[-600, 600]` |
| Relative checkpoint position | `(2,)` | (x, y) of the active checkpoint in the car's local frame, range `[-600, 600]` |
| LiDAR readings | `(36,)` | Normalised distances to obstacles at 36 evenly-spaced angles around the car, range `[0, 1]` |
 
## Episode Termination
 
An episode ends when any of the following occur:
 
- The car **reaches the main goal** (within 1.5 units)
- The car **collides** with any object other than the ground plane
- The **step limit** of 50,000 physics sub-steps is exceeded

## Constructor Parameters
 
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `isDiscrete` | bool | `True` | Use discrete (9-action) or continuous action space |
| `renders` | bool | `False` | Launch PyBullet GUI for visualisation |
| `minimum_safe_distance` | float | `1.0` | Minimum distance to obstacles before a safety violation (currently informational) |
| `reward_callback` | callable | `None` | External function that computes the step reward (required) |
| `observation_callback` | callable | `None` | External function that builds the observation vector (required) |
| `environment_map` | str | `None` | Path to the `.npz` point cloud file |
| `checkpoint_frequency` | int | `10` | How many A* path nodes to skip between spawned checkpoints |
 
<hr style="border: 5px solid #fdb100;">

# AI Car Training and Testing

The car is trained using **Proximal Policy Optimization (PPO)** via the [Stable Baselines3](https://stable-baselines3.readthedocs.io/) library, inside a custom PyBullet simulation environment built from the segmented point cloud data.

## How Training Works
 
Training is handled by `train.py`, which sets up multiple parallel environments using `SubprocVecEnv` and runs PPO to learn a driving policy end-to-end.
 
**Observation Space** — At each step the car receives:
 
- The relative (x, y) position of the **main goal** in the car's local frame
- The relative (x, y) position of the **nearest checkpoint** in the car's local frame (falls back to the goal position if no checkpoint is active)
- A full array of **normalised LiDAR readings** representing distances to surrounding obstacles
**Reward Function** — The reward signal is shaped by several components:
 
| Signal | Value | Condition |
| :--- | :--- | :--- |
| Step penalty | `-0.2` | Every timestep (encourages efficiency) |
| Goal reached | `+200.0` | Car arrives at the main end goal |
| Checkpoint reached | `+150.0` | Car passes through an intermediate checkpoint |
| Progress toward goal | `+10 x delta_distance` | Proportional to distance closed per step |
| Progress toward checkpoint | `+10 x delta_distance` | Proportional to distance closed per step |
| LiDAR proximity penalty | scaled by `-3.0` | Any LiDAR reading below `0.03` (close to obstacle) |
| LiDAR danger penalty | scaled by `-12.0` | Any LiDAR reading below `0.02` (very close to obstacle) |
| Collision penalty | `-400.0` | Car physically collides with an obstacle |
 
Checkpoints spawn along the route at a configurable frequency (`checkpoint_freq`) to provide dense intermediate rewards that guide the car along the correct path rather than taking shortcuts.
 
**Model Checkpoints** — During training, the model is saved every 100,000 timesteps (adjusted for the number of parallel environments) to `model/checkpoints/`. The final model and VecNormalize statistics are saved to `model/` on completion.
 
!!! note "Resuming Training"
    If a model already exists at the specified `model_path`, training will resume from that checkpoint with any updated hyperparameters applied automatically.
 
!!! note "Hardware"
    Training runs on CPU by default. GPU acceleration via CUDA is detected and reported at startup but PPO is configured for CPU. Increase `n_envs` to make better use of multi-core machines.
 
---

## How Testing Works
 
Testing is handled by `test_policy` in `test.py`. It loads a saved PPO checkpoint, launches a rendering-enabled environment, and runs the car through **three fixed obstacle scenarios** in sequence:
 
| Scenario | Description |
| :--- | :--- |
| `midpoint` | Obstacle placed at the midpoint of the route |
| `none` | No obstacles — clean baseline run |
| `random_pos` | Obstacle placed at a random position along the route |
 
The total accumulated reward for each episode is printed at the end of each scenario, giving a structured evaluation across all required test cases.
 
---
 
## Training and Testing Usage Guide
 
**To train the model**, run via `main.py`:
```bash
python src\main.py
```
 
Key training parameters are configured via the `GLOBAL CONFIGURATION PARAMETERS` block in `main.py`. See the [Full System Usage Guide](#usage-guide) above for the full parameter list.
 
**To test a saved model**, run `test.py`:
```bash
python src\test.py
```
 
The model path, point cloud, and checkpoint frequency can be overridden by editing the `test_policy()` call at the bottom of `test.py`:
 
```python
test_policy(
    checkpoint_freq=40,
    model_path="model/checkpoints/ppo_driving_13700000_steps.zip",
    data_path="pointclouds/1_Denoise_NoVeg_Subsampled_centroid.npz"
)
```

<hr style="border: 5px solid #00cd0e;">

# API Reference

<hr style="border: 2px solid #0071ea;">

## clustering.py

??? info "Click to expand this section"

    ### main {: #api_cluster_main }
    ::: clustering.main

    ---

    ### load_laz {: #api_load }
    ::: clustering.load_laz

    ---

    ### visualize {: #api_visualise }
    ::: clustering.visualize

    ---

    ### DavidBentleyScan {: #api_dbscan }
    ::: clustering.DavidBentleyScan

    ---

    ### FelicityRandomForest {: #api_forest }
    ::: clustering.FelicityRandomForest

    ---

    ### CericCentroid {: #api_centroid }
    ::: clustering.CedricCentroid

<hr style="border: 2px solid #0071ea;">
<hr style="border: 2px solid #ff53fc90;">

## train.py
 
??? info "Click to expand this section"
 
    ### custom_observation {: #api_custom_observation }
 
    Builds the observation vector passed to the PPO policy at each timestep.
 
    ```python
    custom_observation(client, car_pos, car_orn, goal_pos_1, goal_orn_1,
                       checkpoint_pos, checkpoint_orn, lidar_readings)
    ```
 
    **Parameters:**
 
    | Name | Type | Description |
    | :--- | :--- | :--- |
    | `client` | pybullet client | Active PyBullet physics client used for coordinate transforms |
    | `car_pos` | tuple | Current world-space position of the car `(x, y, z)` |
    | `car_orn` | tuple | Current world-space orientation of the car as a quaternion |
    | `goal_pos_1` | tuple | World-space position of the primary end goal |
    | `goal_orn_1` | tuple | World-space orientation of the primary end goal |
    | `checkpoint_pos` | tuple or None | World-space position of the active checkpoint, or `None` if no checkpoint is active |
    | `checkpoint_orn` | tuple or None | World-space orientation of the active checkpoint |
    | `lidar_readings` | np.ndarray | Raw normalised LiDAR distances from the car to surrounding obstacles |
 
    **Returns:** `np.ndarray` — A flat observation array of shape `(4 + N,)` where the first 4 elements are the relative (x, y) positions of the goal and checkpoint in the car's local frame, and the remaining N elements are the LiDAR readings.
 
    !!! note
        If `checkpoint_pos` is `None`, the goal position is used as a dummy checkpoint value so the observation shape stays consistent.
 
    ---
 
    ### custom_reward {: #api_custom_reward }
 
    Computes the scalar reward for a single environment step.
 
    ```python
    custom_reward(car_pos, goal_pos_1, checkpoint_pos, lidar_readings,
                  prev_dist_to_goal_1, prev_dist_to_checkpoint,
                  dist_to_goal_1, dist_to_checkpoint,
                  reached_goal_1, reached_checkpoint, collided)
    ```
 
    **Parameters:**
 
    | Name | Type | Description |
    | :--- | :--- | :--- |
    | `car_pos` | tuple | Current world-space position of the car |
    | `goal_pos_1` | tuple | World-space position of the primary end goal |
    | `checkpoint_pos` | tuple or None | World-space position of the active checkpoint |
    | `lidar_readings` | np.ndarray | Normalised LiDAR distances (same as passed to `custom_observation`) |
    | `prev_dist_to_goal_1` | float or None | Distance to the goal at the previous step |
    | `prev_dist_to_checkpoint` | float or None | Distance to the checkpoint at the previous step |
    | `dist_to_goal_1` | float | Distance to the goal at the current step |
    | `dist_to_checkpoint` | float or None | Distance to the checkpoint at the current step, or `None` if inactive |
    | `reached_goal_1` | bool | Whether the car reached the primary goal this step |
    | `reached_checkpoint` | bool | Whether the car reached the active checkpoint this step |
    | `collided` | bool | Whether the car collided with an obstacle this step |
 
    **Returns:** `float` — The total scalar reward for this step.
 
    **Reward breakdown:**
 
    | Component | Value |
    | :--- | :--- |
    | Step penalty | `-0.2` per step |
    | Goal reached | `+200.0` |
    | Checkpoint reached | `+150.0` |
    | Progress reward (goal) | `+10.0 × (prev_dist - cur_dist)` |
    | Progress reward (checkpoint) | `+10.0 × (prev_dist - cur_dist)` |
    | LiDAR proximity penalty | `-3.0 × (0.03 - min_lidar)` when `min_lidar < 0.03` |
    | LiDAR danger penalty | `-12.0 × (0.02 - min_lidar)` when `min_lidar < 0.02` |
    | Collision penalty | `-400.0` |
 
    ---
 
    ### run_training {: #api_run_training }
 
    Sets up the training environment and runs the PPO training loop.
 
    ```python
    run_training(checkpoint_freq, model_path, total_timesteps, n_envs, n_steps,
                 batch_size, n_epochs, learning_rate, entropy_coef, gae_lambda,
                 gamma, max_grad_norm, clip_range,
                 data_path="pointclouds/1_Denoise_NoVeg_Subsampled_centroid.npz") -> str
    ```
 
    **Parameters:**
 
    | Name | Type | Description |
    | :--- | :--- | :--- |
    | `checkpoint_freq` | int | Frequency at which intermediate goal checkpoints spawn in the environment |
    | `model_path` | str | Path to an existing model checkpoint to resume from, or a new path to save to |
    | `total_timesteps` | int | Total number of environment timesteps to train for |
    | `n_envs` | int | Number of parallel environments (uses `SubprocVecEnv`) |
    | `n_steps` | int | Steps collected per environment per PPO rollout |
    | `batch_size` | int | Minibatch size for PPO gradient updates |
    | `n_epochs` | int | Number of epochs per PPO update |
    | `learning_rate` | float | Optimizer learning rate |
    | `entropy_coef` | float | Entropy bonus coefficient |
    | `gae_lambda` | float | Lambda for Generalized Advantage Estimation |
    | `gamma` | float | Reward discount factor |
    | `max_grad_norm` | float | Gradient clipping threshold |
    | `clip_range` | float | PPO policy update clipping range |
    | `data_path` | str | Path to the `.npz` point cloud file used to build the environment |
 
    **Returns:** `str` — Path to the saved final model (without `.zip` extension).
 
<hr style="border: 2px solid #0071ea;">
<hr style="border: 2px solid #ff53fc90;">

## test.py
 
??? info "Click to expand this section"
 
    ### test_policy {: #api_test_policy }
 
    Loads a saved PPO model and evaluates it across three required obstacle scenarios with rendering enabled.
 
    ```python
    test_policy(checkpoint_freq=10,
                model_path="model/checkpoints/ppo_driving_13700000_steps",
                data_path="pointclouds/1_Denoise_NoVeg_Subsampled_centroid.npz")
    ```
 
    **Parameters:**
 
    | Name | Type | Description |
    | :--- | :--- | :--- |
    | `checkpoint_freq` | int | Frequency at which intermediate goal checkpoints spawn in the test environment. Should match the value used during training. Default: `10` |
    | `model_path` | str | Relative path to the saved PPO model checkpoint. The `.zip` extension is optional — it will be appended automatically if missing. Default: `"model/checkpoints/ppo_driving_13700000_steps"` |
    | `data_path` | str | Relative path to the `.npz` point cloud file used to build the test environment. Default: `"pointclouds/1_Denoise_NoVeg_Subsampled_centroid.npz"` |
 
    **Scenarios run:**
 
    The model is tested in 3 seperate scenarios each with a randomly generated environment. Each time the environment is reset, the policy runs deterministically (`deterministic=True`) until `terminated` or `truncated` is set, and the total episode reward is printed to stdout.
 
<hr style="border: 2px solid #ff53fc90;">

## Some other python file

??? info "Click to expand this section"
    Hidden text

<hr style="border: 2px solid #ff53fc90;">

