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
 
Key training parameters are configured via the `GLOBAL CONFIGURATION PARAMETERS` block in `main.py`. See the [Full System Usage Guide](index.md#usage-guide) above for the full parameter list.
 
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