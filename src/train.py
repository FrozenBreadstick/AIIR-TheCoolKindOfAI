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
OBSTACLE_PENALTY = -50.0
LIDAR_PENALTY_SCALE = -5.0
GOAL_REWARD = 1000.0
STEP_PENALTY = -0.5
PROGRESS_REWARD_SCALE = 10.0
MINIMUM_SAFE_DISTANCE = 2.0

def custom_observation(client, car_pos, car_orn, goal_pos, goal_orn,
                        lidar_readings):

   
#     observation = [0.0, 0.0] # Dummy return, replace this
    
#     #invert car transform
#     inv_car_pos, inv_car_orn = client.invertTransform(car_pos, car_orn) 
    
#     #relative goal position
#     rel_goal_pos, _ = client.multiplyTransforms(inv_car_pos, inv_car_orn, goal_pos, goal_orn)

#     #relative obstacle position (if it exists)
#     if has_obstacle:
#         rel_obstacle_pos, _ = client.multiplyTransforms(inv_car_pos, inv_car_orn, obstacle_pos + (0.0,), (0.0, 0.0, 0.0, 1.0))
#     else:
#         rel_obstacle_pos = (0.0, 0.0, 0.0)

#     # fill in observation array
    observation[0] = rel_goal_pos[0] # relative x position of the goal
    observation[1] = rel_goal_pos[1] # relative y position of the goal
#     # observation[2] = rel_obstacle_pos[0] # relative x position of the obstacle (or 0 if no obstacle)
#     # observation[3] = rel_obstacle_pos[1] # relative y position of the obstacle (or 0 if no obstacle)
#     # observation[4] = has_obstacle #1.0 if has_obstacle else 0.0 # binary flag indicating

#     #lidar_readings = lidar_readings[::10]     # 360 → 36
#     #lidar_readings = lidar_readings.reshape(36, 10).mean(axis=1) # optional: downsample by averaging every 10 readings into 36 total readings
#     #lidar_readings = lidar_readings / np.max(lidar_readings)    # normalize to [0,1]
    lidar_readings = lidar_readings / 100.0
#     # if int(time.time()) % 2 == 0:  # every ~2 seconds
#     #     print("LIDAR min:", np.min(lidar_readings))


    observation = np.concatenate([observation, lidar_readings])

    return observation

    # redundant stuff below




def custom_reward(car_pos, goal_pos,
                  lidar_readings, prev_dist_to_goal, dist_to_goal, reached_goal):

    reward = 0.0

    reward += STEP_PENALTY   # punish every step so it learns to be efficient

    reward += PROGRESS_REWARD_SCALE * (prev_dist_to_goal - dist_to_goal)
    # positive when it moved closer, negative when it moved away

    if reached_goal:
        reward += GOAL_REWARD   # big reward for success

    if has_obstacle and obstacle_pos is not None:
        dist_to_obstacle = math.sqrt(
            (car_pos[0] - obstacle_pos[0])**2 +
            (car_pos[1] - obstacle_pos[1])**2
        )
        if dist_to_obstacle < MINIMUM_SAFE_DISTANCE:
            # penalty grows the closer it gets (0 at safe distance, full at contact)
            proximity_ratio = (MINIMUM_SAFE_DISTANCE - dist_to_obstacle) / MINIMUM_SAFE_DISTANCE
            reward += OBSTACLE_PENALTY * proximity_ratio

    min_lidar = np.min(lidar_readings)
    if min_lidar < 0.2:   # if any beam is very close to a wall
        reward += LIDAR_PENALTY_SCALE * (0.2 - min_lidar)

    return reward

# You can change these variables for more training steps or if you have a powerful CPU:
TOTAL_TIMESTEPS = 500_000      # define the number of steps used during the training
N_ENVS = 8                   # number of processor core used for multithreading
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

    # ========================================================
    # TODO: Implement PPO using stable_baselines3!
    # 1. Instantiate the PPO agent ("MlpPolicy")
    #    HINT: SB3's default PPO parameters are optimized for long tasks. 
    #    For our short driving environment, training will be painfully slow
    #    unless you override these hyperparameters during instantiation:
    #      - learning_rate=0.0003
    #      - n_steps=512
    #      - batch_size=256
    #      - ent_coef=0.01
    #    You can play around with different parameters, change the number of
    #    TOTAL_TIMESTEPS, learning_rate, etc.
    # 2. Tell the agent to log metrics to a local tensorboard directory.
    # 3. Call agent.learn(total_timesteps=TOTAL_TIMESTEPS)
    # 4. Save the agent when done
    # 
    # Optional: to speed up the training and avoiding to start from scratch every time, 
    # you can reload previously trained models 
    # (look up Curriculum Learning/Transfer Learning to learn more about this)
    # 
    # If you do, keep track of the previous reward function you used for the VIVA 
    # (or retrain from scratch to make sure your function works properly)
    # ========================================================
    #instantiate PPO agent
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
        reset_num_timesteps=True,   # keeps step count when resuming
    )

    # Step 6: save the final model
    os.makedirs("model", exist_ok=True)
    ppo_agent.save(MODEL_PATH)
    print(f"Agent saved to {MODEL_PATH}")

  