import sys
sys.path.append('..')
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import simple_driving
import time
import os
import math
import numpy as np

# ========================================================
# Reward Function Configuration Parameters
# ========================================================
LIDAR_PENALTY_SCALE = -15.0
GOAL_REWARD_1 = 200.0   # closest goal - biggest reward to prioritise it first
GOAL_REWARD_2 = 400.0    # middle goal
GOAL_REWARD_3 = 1000.0    # farthest goal
STEP_PENALTY = -0.1
PROGRESS_REWARD_SCALE_1 = 10.0  # strongest pull toward goal 1 (closest)
PROGRESS_REWARD_SCALE_2 = 7.5
PROGRESS_REWARD_SCALE_3 = 5.0
LIDAR_CLOSE_THRESHOLD = 0.08     # 10 meters
LIDAR_DANGER_THRESHOLD = 0.03   # 5 meters
COLLISION_PENALTY = -200.0


def custom_observation(client, car_pos, car_orn,
                        goal_pos_1, goal_orn_1,
                        goal_pos_2, goal_orn_2,
                        goal_pos_3, goal_orn_3,
                        lidar_readings):

    # 6 values: relative x,y for each of 3 goals
    observation = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    inv_car_pos, inv_car_orn = client.invertTransform(car_pos, car_orn)

    rel_goal_pos_1, _ = client.multiplyTransforms(inv_car_pos, inv_car_orn, goal_pos_1, goal_orn_1)
    rel_goal_pos_2, _ = client.multiplyTransforms(inv_car_pos, inv_car_orn, goal_pos_2, goal_orn_2)
    rel_goal_pos_3, _ = client.multiplyTransforms(inv_car_pos, inv_car_orn, goal_pos_3, goal_orn_3)

    observation[0] = rel_goal_pos_1[0]
    observation[1] = rel_goal_pos_1[1]
    observation[2] = rel_goal_pos_2[0]
    observation[3] = rel_goal_pos_2[1]
    observation[4] = rel_goal_pos_3[0]
    observation[5] = rel_goal_pos_3[1]

    # Normalise lidar to [0, 1]
    lidar_normalised = lidar_readings / 100.0

    observation = np.concatenate([observation, lidar_normalised])

    return observation


def custom_reward(car_pos, goal_pos_1, goal_pos_2, goal_pos_3,
                  lidar_readings,
                  prev_dist_to_goal_1, prev_dist_to_goal_2, prev_dist_to_goal_3,
                  dist_to_goal_1, dist_to_goal_2, dist_to_goal_3,
                  reached_goal_1, reached_goal_2, reached_goal_3,
                  goal_1_reward_given, goal_2_reward_given, goal_3_reward_given,
                  collided):

    reward = 0.0

    # Step penalty to encourage efficiency
    reward += STEP_PENALTY

    # --- Goal rewards (one-time each) ---
    if reached_goal_1:
        reward += GOAL_REWARD_1
    if reached_goal_2:
        reward += GOAL_REWARD_2
    if reached_goal_3:
        reward += GOAL_REWARD_3

    # --- Progress reward toward the next uncollected goal ---
    # Prioritise closest goal first, then middle, then far
    if not goal_1_reward_given:
        progress = np.clip(prev_dist_to_goal_1 - dist_to_goal_1, -2.0, 2.0)
        reward += PROGRESS_REWARD_SCALE_1 * progress
    elif not goal_2_reward_given:
        progress = np.clip(prev_dist_to_goal_2 - dist_to_goal_2, -2.0, 2.0)
        reward += PROGRESS_REWARD_SCALE_2 * progress
    elif not goal_3_reward_given:
        progress = np.clip(prev_dist_to_goal_3 - dist_to_goal_3, -2.0, 2.0)
        reward += PROGRESS_REWARD_SCALE_3 * progress

    # --- Proximity hint: small reward for being close to any uncollected goal ---
    max_dist = 400.0
    if not goal_1_reward_given:
        reward += (1.0 - dist_to_goal_1 / max_dist) * 2.0
    elif not goal_2_reward_given:
        reward += (1.0 - dist_to_goal_2 / max_dist) * 2.0
    elif not goal_3_reward_given:
        reward += (1.0 - dist_to_goal_3 / max_dist) * 2.0

    # --- LiDAR wall avoidance ---
    normalised_lidar = lidar_readings / 100.0
    min_lidar = np.min(normalised_lidar)

    if min_lidar < LIDAR_CLOSE_THRESHOLD:
        reward += LIDAR_PENALTY_SCALE * (LIDAR_CLOSE_THRESHOLD - min_lidar)

    if min_lidar < LIDAR_DANGER_THRESHOLD:
        reward += LIDAR_PENALTY_SCALE * 3.0 * (LIDAR_DANGER_THRESHOLD - min_lidar)

    # --- Collision penalty ---
    if collided:
        reward += COLLISION_PENALTY
        return reward  # return immediately

    return reward


# Training configuration
TOTAL_TIMESTEPS = 5_000_000
N_ENVS = 8
MODEL_PATH = "model/ppo_simple_driving_model"

if __name__ == "__main__":
    env_kwargs = {
        "renders": False,
        "isDiscrete": False,
        "reward_callback": custom_reward,
        "observation_callback": custom_observation,
        "environment_map": r"pointclouds\1_Denoise_NoVeg_Subsampled_centroid.npz",
        "max_steps": 30000
    }
    env = make_vec_env(
        "SimpleDriving-v0",
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=env_kwargs,
        vec_env_kwargs={"start_method": "spawn"}
    )

    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"Loading existing model from {MODEL_PATH} ...")
        ppo_agent = PPO.load(MODEL_PATH, env=env, device="cpu", tensorboard_log="./ppo_tensorboard/")
    else:
        ppo_agent = PPO(
            "MlpPolicy",
            env,
            learning_rate=0.0003,
            n_steps=512,
            batch_size=256,
            ent_coef=0.01,
            verbose=1,
            device="cpu",
            tensorboard_log="./ppo_tensorboard/"
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path="./model/checkpoints/",
        name_prefix="ppo_driving",
    )

    ppo_agent.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_cb,
        reset_num_timesteps=True,
    )

    os.makedirs("model", exist_ok=True)
    ppo_agent.save(MODEL_PATH)
    print(f"Agent saved to {MODEL_PATH}")