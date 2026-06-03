import sys
sys.path.append('..')
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import get_schedule_fn
from pybullet_utils import bullet_client as bc
import simple_driving
import time
import os
import math
import numpy as np
import torch
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))


# ========================================================
# Reward Function Configuration Parameters
# ========================================================
GOAL_REWARD_1 = 200.0   # GOAL REWARD 1 and PROGRESS REWARD SCALE 1 refer to the main goal at the end of the track
GOAL_REWARD_2 = 150.0   # GOAL REWARD 2 and PROGRESS REWARD SCALE 2 refer to the checkpoints that spawn along the track
STEP_PENALTY = -0.2
PROGRESS_REWARD_SCALE_1 = 10.0
PROGRESS_REWARD_SCALE_2 = 10.0
LIDAR_CLOSE_THRESHOLD = 0.03
LIDAR_DANGER_THRESHOLD = 0.02
DANGER_MULTIPLIER = 4.0 
LIDAR_PENALTY_SCALE = -3.0
COLLISION_PENALTY = -400.0  

# ========================================================
# Custom Reward and Observation Callbacks
# ========================================================
def custom_observation(
    client: bc.BulletClient,
    car_pos: tuple,
    car_orn: tuple,
    goal_pos_1: tuple,
    goal_orn_1: tuple,
    checkpoint_pos: tuple | None,
    checkpoint_orn: tuple | None,
    lidar_readings: np.ndarray
) -> np.ndarray:
    """
    Constructs the custom observation array for the environment.

    Args:
        client: Active PyBullet physics client used for coordinate transforms
        car_pos: Current world-space position of the car `(x, y, z)`
        car_orn: Current world-space orientation of the car as a quaternion
        goal_pos_1: World-space position of the primary end goal
        goal_orn_1: World-space orientation of the primary end goal
        checkpoint_pos: World-space position of the active checkpoint, or `None` if no checkpoint is active
        checkpoint_orn: World-space orientation of the active checkpoint
        lidar_readings: Raw normalised LiDAR distances from the car to surrounding obstacles

    Returns:
        A flat observation array of shape `(4 + N,)` where the first 4 elements are the relative (x, y) positions of the goal and checkpoint in the car's local frame, and the remaining N elements are the LiDAR readings.
    """
    observation = [0.0, 0.0, 0.0, 0.0] # placeholder for relative goal position (x, y) of the 3 goals

    if checkpoint_pos is None:
        checkpoint_pos = goal_pos_1  # if no checkpoint, use goal position as dummy
        checkpoint_orn = goal_orn_1

    # invert car transform
    inv_car_pos, inv_car_orn = client.invertTransform(car_pos, car_orn)

    # relative goal position
    rel_goal_pos_1, _ = client.multiplyTransforms(inv_car_pos, inv_car_orn, goal_pos_1, goal_orn_1)

    rel_checkpoint_pos, _ = client.multiplyTransforms(
        inv_car_pos, inv_car_orn,
        checkpoint_pos, checkpoint_orn
    )
    
    observation[0] = rel_goal_pos_1[0]
    observation[1] = rel_goal_pos_1[1]
    observation[2] = rel_checkpoint_pos[0]
    observation[3] = rel_checkpoint_pos[1]

    # normalise lidar to [0, 1]
    # lidar_readings = lidar_readings / 100.0
    # convert LiDAR readings to relative positions of detected obstacles in the car's local frame
    # (assuming lidar_readings are distances at fixed angles around the car)
    # lidar_positions = []
    # for i, distance in enumerate(lidar_readings):
    #     angle = 2 * math.pi * i / len(lidar_readings)  # angle of this LiDAR ray
    #     x = 100 * distance * math.cos(angle)  # convert polar to Cartesian coordinates
    #     y = 100 * distance * math.sin(angle)
    #     lidar_positions.append(x)  # add both x and y to the observation
    #     lidar_positions.append(y)

    # lidar_positions = np.array(lidar_positions, dtype=np.float32)

    observation = np.concatenate([observation, lidar_readings.astype(np.float32)])

    return observation


def custom_reward(
    car_pos: tuple,
    goal_pos_1: tuple,
    checkpoint_pos: tuple | None,
    lidar_readings: np.ndarray,
    prev_dist_to_goal_1: float | None,
    prev_dist_to_checkpoint: float | None,
    dist_to_goal_1: float,
    dist_to_checkpoint: float | None,
    reached_goal_1: bool,
    reached_checkpoint: bool,
    collided: bool
) -> float:
    """
    Calculates the environment reward for the current step.

    Args:
        car_pos: Current world-space position of the car
        goal_pos_1: World-space position of the primary end goal
        checkpoint_pos: World-space position of the active checkpoint
        lidar_readings: Normalised LiDAR distances (same as passed to `custom_observation`)
        prev_dist_to_goal_1: Distance to the goal at the previous step
        prev_dist_to_checkpoint: Distance to the checkpoint at the previous step
        dist_to_goal_1: Distance to the goal at the current step
        dist_to_checkpoint: Distance to the checkpoint at the current step, or `None` if inactive
        reached_goal_1: Whether the car reached the primary goal this step
        reached_checkpoint: Whether the car reached the active checkpoint this step
        collided: Whether the car collided with an obstacle this step

    Returns:
        The total scalar reward for this step.
    """
    reward = 0.0

    # step penalty
    reward += STEP_PENALTY

    # reward for making progress toward the goals, calculated as the change in distance to each goal since the last step
    # big reward for reaching the goal small for on the way
    if reached_goal_1:
        reward += GOAL_REWARD_1
    elif prev_dist_to_goal_1 is not None and dist_to_checkpoint is None:
        reward += PROGRESS_REWARD_SCALE_1 * (prev_dist_to_goal_1 - dist_to_goal_1)
    
    if reached_checkpoint:
        reward += GOAL_REWARD_2
    elif (dist_to_checkpoint is not None
          and prev_dist_to_checkpoint is not None):
        reward += PROGRESS_REWARD_SCALE_2 * (prev_dist_to_checkpoint - dist_to_checkpoint)
 

    # ---- wall / obstacle avoidance via lidar ----
    # normalised_lidar = lidar_readings / 100 # already normalised
    min_lidar = np.min(lidar_readings)
    # print(f"Minimum LiDAR reading: {min_lidar:.3f}")

    if min_lidar < LIDAR_CLOSE_THRESHOLD:
        reward += LIDAR_PENALTY_SCALE * (LIDAR_CLOSE_THRESHOLD - min_lidar)

    if min_lidar < LIDAR_DANGER_THRESHOLD:
        reward += DANGER_MULTIPLIER * LIDAR_PENALTY_SCALE * (LIDAR_DANGER_THRESHOLD - min_lidar)
    
    if collided:
        reward += COLLISION_PENALTY  # strong enough to outweigh the shortcut

    return reward

# ========================================================
# Training Loop
# ========================================================

# You can change these variables for more training steps or if you have a powerful CPU:
def run_training(
    checkpoint_freq: int,
    model_path: str,
    total_timesteps: int,
    n_envs: int,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
    entropy_coef: float,
    gae_lambda: float,
    gamma: float,
    max_grad_norm: float,
    clip_range: float,
    data_path: str = "pointclouds/1_Denoise_NoVeg_Subsampled_centroid.npz"
) -> str:
    """
    The Training function that sets up the environment and trains either a new PPO model or loads an existing one to build upon.

    Args:
        checkpoint_freq: Frequency at which intermediate goal checkpoints spawn in the environment
        model_path: Path to an existing model checkpoint to resume from, or a new path to save to
        data_path: Path to the `.npz` point cloud file used to build the environment
        total_timesteps: Total number of environment timesteps to train for
        n_envs: Number of parallel environments (uses `SubprocVecEnv`)
        n_steps: Steps collected per environment per PPO rollout
        batch_size: Minibatch size for PPO gradient updates
        n_epochs: Number of epochs per PPO update
        learning_rate: Optimizer learning rate
        entropy_coef: Entropy bonus coefficient
        gae_lambda: Lambda for Generalized Advantage Estimation
        gamma: Reward discount factor
        max_grad_norm: Gradient clipping threshold
        clip_range: PPO policy update clipping range

    Returns:
        Path to the saved final model (without `.zip` extension).
    """
    env_kwargs = {
        "checkpoint_frequency": checkpoint_freq,
        "renders": False,
        "isDiscrete": False,
        "reward_callback": custom_reward,
        "observation_callback": custom_observation,
        "environment_map": data_path
    }
    env = make_vec_env(
        "SimpleDriving-v0",
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=env_kwargs,
        vec_env_kwargs={"start_method": "spawn"}
    )

    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    if os.path.exists(model_path + ".zip"):
        print(f"Loading existing model from {model_path} ...")
        ppo_agent = PPO.load(model_path, env=env, device="cpu", tensorboard_log="./ppo_tensorboard/")             
        # Override saved hyperparameters with current values
        # PPO.load restores whatever was saved with the checkpoint,
        # so without this your constants at the top have no effect on resumed runs
        ppo_agent.learning_rate = get_schedule_fn(learning_rate)
        ppo_agent.ent_coef = entropy_coef
        ppo_agent.clip_range = get_schedule_fn(clip_range)
        ppo_agent.max_grad_norm = max_grad_norm
        ppo_agent.n_epochs = n_epochs
        ppo_agent.gamma = gamma
        ppo_agent.gae_lambda = gae_lambda
        ppo_agent.set_env(env)
        print(f"Hyperparameters updated: lr={learning_rate}, ent_coef={entropy_coef}, gamma={gamma}, gae_lambda={gae_lambda}, clip_range={clip_range}, max_grad_norm={max_grad_norm}, n_epochs={n_epochs}")
    else:
        ppo_agent = PPO(
            "MlpPolicy",
            env,
            # --- core params ---
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            # --- Discount and advantage ---
            gamma=gamma,
            gae_lambda=gae_lambda,
            # --- stability ---
            clip_range=clip_range,
            max_grad_norm=max_grad_norm,
            # --- exploration ---
            ent_coef=entropy_coef,
            verbose=1,
            device="cpu",
            tensorboard_log="./ppo_tensorboard/"
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1),
        save_path="./model/checkpoints/",
        name_prefix="ppo_driving",
    )

    ppo_agent.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        reset_num_timesteps=not os.path.exists(model_path + ".zip"),
    )

    model_step_count = ppo_agent.num_timesteps

    os.makedirs("model", exist_ok=True)
    ppo_agent.save("model/ppo_driving_{}_steps".format(model_step_count))
    env.save(os.path.join("model", "vecnormalize.pkl"))
    print(f"Agent saved to model/ppo_driving_{model_step_count}_steps.zip and VecNormalize stats saved to model/vecnormalize.pkl")

    return ("model/ppo_driving_{}_steps".format(model_step_count))

if __name__ == "__main__":
    run_training()