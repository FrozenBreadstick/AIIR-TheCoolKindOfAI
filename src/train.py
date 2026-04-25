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
LIDAR_PENALTY_SCALE = -20.0
GOAL_REWARD = 1000.0
STEP_PENALTY = -2.0
PROGRESS_REWARD_SCALE = 10.0
MINIMUM_SAFE_DISTANCE = 2.0

def custom_observation(client, car_pos, car_orn, goal_pos, goal_orn, obstacle_pos, has_obstacle, lidar_readings):
    """
    Computes the observation array for the neural network.
    
    Args:
        client (bullet_client): The PyBullet physics client.
        car_pos (list of float): The global [x, y, z] position of the car.
        car_orn (list of float): The global [x, y, z, w] quaternion orientation of the car.
        goal_pos (list of float): The global [x, y, z] position of the goal.
        goal_orn (list of float): The global [x, y, z, w] quaternion orientation of the goal.
        obstacle_pos (tuple of float or None): The global (x, y) position of the obstacle, if it exists.
        has_obstacle (bool): True if an obstacle spawned this episode, False otherwise.
        lidar_readings (numpy.ndarray): The LiDAR readings.

    Returns:
        list of float: The computed observation state array.
    """
    # ========================================================
    # TODO: Calculate the Observation Space for the Neural Network
    # By default, PyBullet returns global coordinates (X, Y).
    # You must convert the goal position and obstacle position into 
    # RELATIVE coordinates (where is the object relative to the car?)
    # HINT: Look up client.invertTransform and client.multiplyTransforms
    # ========================================================
    
    observation = [0.0, 0.0, 0.0, 0.0, 0.0] # Dummy return, replace this
    
    #invert car transform
    inv_car_pos, inv_car_orn = client.invertTransform(car_pos, car_orn) 
    
    #relative goal position
    rel_goal_pos, _ = client.multiplyTransforms(inv_car_pos, inv_car_orn, goal_pos, goal_orn)

    #relative obstacle position (if it exists)
    if has_obstacle:
        rel_obstacle_pos, _ = client.multiplyTransforms(inv_car_pos, inv_car_orn, obstacle_pos + (0.0,), (0.0, 0.0, 0.0, 1.0))
    else:
        rel_obstacle_pos = (0.0, 0.0, 0.0)

    # fill in observation array
    observation[0] = rel_goal_pos[0] # relative x position of the goal
    observation[1] = rel_goal_pos[1] # relative y position of the goal
    observation[2] = rel_obstacle_pos[0] # relative x position of the obstacle (or 0 if no obstacle)
    observation[3] = rel_obstacle_pos[1] # relative y position of the obstacle (or 0 if no obstacle)
    observation[4] = has_obstacle #1.0 if has_obstacle else 0.0 # binary flag indicating

    #lidar_readings = lidar_readings[::10]     # 360 → 36
    #lidar_readings = lidar_readings.reshape(36, 10).mean(axis=1) # optional: downsample by averaging every 10 readings into 36 total readings
    lidar_readings = lidar_readings / np.max(lidar_readings)    # normalize to [0,1]

    if int(time.time()) % 2 == 0:  # every ~2 seconds
        print("LIDAR min:", np.min(lidar_readings))

    observation = np.concatenate([observation, lidar_readings])

    return observation


def custom_reward(car_pos, goal_pos, obstacle_pos, has_obstacle, lidar_readings, prev_dist_to_goal, dist_to_goal, reached_goal):
    """
    Computes the scalar reward for the current timestep.
    
    Args:
        car_pos (list of float): The global [x, y, z] position of the car.
        goal_pos (list of float): The global [x, y, z] position of the goal.
        obstacle_pos (tuple of float or None): The global (x, y) position of the obstacle, if it exists.
        has_obstacle (bool): True if an obstacle spawned this episode.
        lidar_readings (numpy.ndarray): The LiDAR readings.
        prev_dist_to_goal (float): The distance to the goal in the previous physics frame.
        dist_to_goal (float): The distance to the goal in the current physics frame.
        reached_goal (bool): True if the car reached the goal this frame.
        
    Returns:
        float: The exact mathematical reward for this timestep.
    """
    # ========================================================
    # TODO: Write your reward function
    # 1. Give the agent a basic STEP_PENALTY every frame
    # 2. Reward it for getting closer to the goal
    # 3. Give it a large GOAL_REWARD if it reached_goal
    # 4. Give it a large OBSTACLE_PENALTY if it gets too close to the obstacle
    # 5. Give it a penalty for close LiDAR readings
    # 
    # HINT: If your agent has trouble avoiding the obstacle and drives right into it,
    # you can try adding a "proximity penalty" (repulsive field). If the car gets 
    # within a certain distance of the obstacle, start gradually subtracting reward!
    # ========================================================
    
    reward = 0.0 # Dummy return, replace this

    # basic step penalty
    reward += STEP_PENALTY

    # reward for progress towards the goal
    reward += PROGRESS_REWARD_SCALE * (prev_dist_to_goal - dist_to_goal) # (positive if closer, negative if further)

    # reward for reaching the goal
    if reached_goal:
        reward += GOAL_REWARD

    # penalty for being too close to the obstacle
    if has_obstacle:
        dist_to_obstacle = math.sqrt((car_pos[0] - obstacle_pos[0])**2 + (car_pos[1] - obstacle_pos[1])**2)
        if dist_to_obstacle < MINIMUM_SAFE_DISTANCE:
            reward += OBSTACLE_PENALTY * (MINIMUM_SAFE_DISTANCE - dist_to_obstacle / MINIMUM_SAFE_DISTANCE) # more penalty the closer it is

    if np.min(lidar_readings) < 0.2: # if any LiDAR reading is very close to an obstacle
        reward += LIDAR_PENALTY_SCALE * (0.2 - np.min(lidar_readings)) # more penalty the closer it is

    return reward

# You can change these variables for more training steps or if you have a powerful CPU:
TOTAL_TIMESTEPS = 750      # define the number of steps used during the training
N_ENVS = 4                   # number of processor core used for multithreading

if __name__ == "__main__":
    env_kwargs = {
        "renders": False, 
        "isDiscrete": False,
        "reward_callback": custom_reward,          
        "observation_callback": custom_observation 
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

    # call agent.learn
    ppo_agent.learn(total_timesteps=TOTAL_TIMESTEPS)

    # save the agent
    ppo_agent.save("model/ppo_simple_driving_model")
    print("----------------------")
    print("----------------------")
    print("----------------------")
    print("agent saved!")
    print("----------------------")
    print("----------------------")
    print("----------------------")

    #print("Dummy script - Implement PPO here.")
