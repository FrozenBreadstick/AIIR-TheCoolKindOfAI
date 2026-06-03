import sys
sys.path.append('..')
import gymnasium as gym
from stable_baselines3 import PPO
import simple_driving
import time
from train import custom_reward, custom_observation

def test_policy(checkpoint_freq = 10, model_path = "model\checkpoints\ppo_driving_13700000_steps.zip"):
    print("Loading saved PPO model...")

    print("Loading environment with rendering enabled...")
    env = gym.make("SimpleDriving-v0", checkpoint_frequency=checkpoint_freq, renders=True, isDiscrete=False, reward_callback=custom_reward, observation_callback=custom_observation, environment_map=r"pointclouds\1_Denoise_NoVeg_Subsampled_centroid.npz")
    #model.set_env(env)
    model = PPO.load(model_path, env=env)

    scenarios = ["midpoint", "none", "random_pos"]
    print(f"Starting evaluation covering the {len(scenarios)} required obstacle scenarios...")

    for ep, scenario in enumerate(scenarios):
        print(f"\n--- Scenario {ep + 1}: {scenario.upper()} ---")
        obs, info = env.reset(options={"scenario": scenario})
        done = False
        episode_reward = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            time.sleep(0.01)
            
        print(f"Episode {ep + 1} finished - Total Reward: {episode_reward:.2f}")

    env.close()

if __name__ == "__main__":
    test_policy(checkpoint_freq=40, model_path="model\checkpoints\ppo_driving_13700000_steps.zip")
