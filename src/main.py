#Imports
import sys
import os
import time
import logging

import clustering
import train
import test

#=======================================================
#========= Global Configuration Parameters =============
#=======================================================

# Test/Train - Checkpoint and Model Configuration
CHECKPOINT_FREQ = 40 # Checkpoint frequency in terms of how many goals spawn in the environment (not PPO update steps)
MODEL_PATH = "tensor\ppo39.40_137\ppo_driving_10700000_steps" # if you ONLY testing a model, include the relative model path you would like to test. Otherwise if you are training a model include the relative model path you would like to start your training from. Delete the .zip file if you want to start fresh.

# Train - Training Configuration Parameters
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

#=======================================================
#=======================================================
#=======================================================

def Test(cluster = True, train = False, test = True, model_path = None, checkpoint_freq = 10, train_params = None):
    if not os.path.exists(model_path + ".zip"):
        print(no_model_found_msg := f"Model checkpoint not found at {model_path}.zip. Training a new model.")
        train = True
    if model_path is None:
        print(no_model_path_msg := "No model path provided. Training a new model.")
        train = True
    
    if cluster:
        clustering.main() 
    if train:
        train.run_training(model_path=model_path, checkpoint_freq=checkpoint_freq, **train_params)
    if test:
        if train == True:
            select_model = input("Enter the model checkpoint you want to test (e.g., 'ppo_driving_13700000_steps'): ")
            select_model = f"model\checkpoints\{select_model}"
        else:
            select_model = model_path
        test.test_policy(checkpoint_freq=checkpoint_freq, model_path=select_model)


if __name__ == "__main__":

    train_params = {
        "total_timesteps": TOTAL_TIMESTEPS,
        "n_envs": N_ENVS,
        "n_steps": N_STEPS,
        "batch_size": BATCH_SIZE,
        "n_epochs": N_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "entropy_coef": ENTROPY_COEF,
        "gae_lambda": GAE_LAMBDA,
        "gamma": GAMMA,
        "max_grad_norm": MAX_GRAD_NORM,
        "clip_range": CLIP_RANGE
    }

    Test(cluster=False, train=False, test=True, model_path=MODEL_PATH, checkpoint_freq=CHECKPOINT_FREQ, train_params=train_params)
