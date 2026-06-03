import sys
sys.path.append('..')
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import get_schedule_fn
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
# Training Configuration Parameters
# ========================================================
TOTAL_TIMESTEPS = 3_000_000
N_ENVS = 8
N_STEPS = 1024
BATCH_SIZE = 512
N_EPOCHS = 4
LEARNING_RATE = 0.0001
ENTROPY_COEF = 0.15
GAE_LAMBDA = 0.95
GAMMA = 0.995
MAX_GRAD_NORM = 0.3
CLIP_RANGE = 0.2

# Checkpoint frequency in terms of how many goals spawn in the environment (not PPO update steps)
CHECKPOINT_FREQ = 40

# include the relative model path you would like to start your training from. Delete the .zip file if you want to start fresh.
MODEL_PATH      = "model\checkpoints\ppo_driving_10700000_steps"

# ========================================================
# Custom Reward and Observation Callbacks
# ========================================================
def custom_observation(client, car_pos, car_orn, goal_pos_1, goal_orn_1, checkpoint_pos, checkpoint_orn, lidar_readings):

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


def custom_reward(car_pos, goal_pos_1, checkpoint_pos, lidar_readings, prev_dist_to_goal_1, prev_dist_to_checkpoint,
                  dist_to_goal_1, dist_to_checkpoint, reached_goal_1, reached_checkpoint, collided):

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
def run_training() -> None:
    env_kwargs = {
        "checkpoint_frequency": CHECKPOINT_FREQ,
        "renders": False,
        "isDiscrete": False,
        "reward_callback": custom_reward,
        "observation_callback": custom_observation,
        "environment_map": r"pointclouds\1_Denoise_NoVeg_Subsampled_centroid.npz"
    }
    env = make_vec_env(
        "SimpleDriving-v0",
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=env_kwargs,
        vec_env_kwargs={"start_method": "spawn"}
    )

    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"Loading existing model from {MODEL_PATH} ...")
        ppo_agent = PPO.load(MODEL_PATH, env=env, device="cpu", tensorboard_log="./ppo_tensorboard/")             
        # Override saved hyperparameters with current values
        # PPO.load restores whatever was saved with the checkpoint,
        # so without this your constants at the top have no effect on resumed runs
        ppo_agent.learning_rate = get_schedule_fn(LEARNING_RATE)
        ppo_agent.ent_coef = ENTROPY_COEF
        ppo_agent.clip_range = get_schedule_fn(CLIP_RANGE)
        ppo_agent.max_grad_norm = MAX_GRAD_NORM
        ppo_agent.n_epochs = N_EPOCHS
        ppo_agent.gamma = GAMMA
        ppo_agent.gae_lambda = GAE_LAMBDA
        ppo_agent.set_env(env)
        print(f"Hyperparameters updated: lr={LEARNING_RATE}, ent_coef={ENTROPY_COEF}, gamma={GAMMA}")
    else:
        ppo_agent = PPO(
            "MlpPolicy",
            env,
            # --- core params ---
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            # --- Discount and advantage ---
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            # --- stability ---
            clip_range=CLIP_RANGE,
            max_grad_norm=MAX_GRAD_NORM,
            # --- exploration ---
            ent_coef=ENTROPY_COEF,
            verbose=1,
            device="cpu",
            tensorboard_log="./ppo_tensorboard/"
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(100_000 // N_ENVS, 1),
        save_path="./model/checkpoints/",
        name_prefix="ppo_driving",
    )

    ppo_agent.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_cb,
        reset_num_timesteps=not os.path.exists(MODEL_PATH + ".zip"),
    )

    os.makedirs("model", exist_ok=True)
    ppo_agent.save(MODEL_PATH)
    env.save(os.path.join("model", "vecnormalize.pkl"))
    print(f"Agent saved to {MODEL_PATH}")

if __name__ == "__main__":
    run_training()