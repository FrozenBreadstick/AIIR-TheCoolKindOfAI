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