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
LIDAR_PENALTY_SCALE = -5.0
GOAL_REWARD = 1000.0
STEP_PENALTY = -0.5
PROGRESS_REWARD_SCALE = 10.0
LIDAR_CLOSE_THRESHOLD = 0.2      # normalised lidar distance considered "too close"
LIDAR_DANGER_THRESHOLD = 0.1     # even closer → harsher penalty

def custom_observation(client, car_pos, car_orn, goal_pos, goal_orn,
                        lidar_readings):

    observation = [0.0, 0.0]

    # invert car transform
    inv_car_pos, inv_car_orn = client.invertTransform(car_pos, car_orn)

    # relative goal position
    rel_goal_pos, _ = client.multiplyTransforms(inv_car_pos, inv_car_orn, goal_pos, goal_orn)

    observation[0] = rel_goal_pos[0]
    observation[1] = rel_goal_pos[1]

    # normalise lidar to [0, 1]
    lidar_readings = lidar_readings / 100.0

    observation = np.concatenate([observation, lidar_readings])

    return observation


def custom_reward(car_pos, goal_pos,
                  lidar_readings, prev_dist_to_goal, dist_to_goal, reached_goal):

    reward = 0.0

    # small penalty every step to encourage efficiency
    reward += STEP_PENALTY

    # reward for making progress toward the goal
    reward += PROGRESS_REWARD_SCALE * (prev_dist_to_goal - dist_to_goal)

    # big reward for reaching the goal
    if reached_goal:
        reward += GOAL_REWARD

    # ---- wall / obstacle avoidance via lidar ----
    normalised_lidar = lidar_readings # already normalised
    min_lidar = np.min(normalised_lidar)

    # graduated penalty: the closer the nearest wall, the larger the penalty
    if min_lidar < LIDAR_CLOSE_THRESHOLD:
        # linear penalty that grows as distance shrinks
        reward += LIDAR_PENALTY_SCALE * (LIDAR_CLOSE_THRESHOLD - min_lidar)

    # extra harsh penalty when extremely close (about to collide)
    if min_lidar < LIDAR_DANGER_THRESHOLD:
        reward += LIDAR_PENALTY_SCALE * 2.0 * (LIDAR_DANGER_THRESHOLD - min_lidar)

    return reward


# You can change these variables for more training steps or if you have a powerful CPU:
TOTAL_TIMESTEPS = 500_000
N_ENVS = 8
MODEL_PATH      = "model/ppo_simple_driving_model"
MAX_GOAL_DIST   = 1200.0

if __name__ == "__main__":
    env_kwargs = {
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

    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"Loading existing model from {MODEL_PATH} ...")
        ppo_agent = PPO.load(MODEL_PATH, env=env, tensorboard_log="./ppo_tensorboard/")
    else:
        ppo_agent = PPO(
            "MlpPolicy",
            env,
            learning_rate=0.0003,
            n_steps=512,
            batch_size=256,
            ent_coef=0.01,
            verbose=1,
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