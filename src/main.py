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

# General Configuration - Mandatory
RUN_CLUSTER = True # Whether to run the clustering script before training/testing. Set to False if you have already run it and have the necessary files in place.
RUN_TRAIN = False # Whether to run the training process. Set to False if you only
RUN_TEST = True # Whether to run the testing process. Set to False if you only want to train a model without testing it immediately after.

CHECKPOINT_FREQ = 30 # Checkpoint frequency in terms of how many goals spawn in the environment (not PPO update steps)
MODEL_PATH = "tensor_data\ppo39.30_2\ppo_driving_6700000_steps" # if you ONLY testing a model, include the relative model path you would like to test. Otherwise if you are training a model include the relative model path you would like to start your training from. Delete the .zip file if you want to start fresh.

# Train - Training Configuration Parameters
TOTAL_TIMESTEPS = 2
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

def Test(run_cluster = True, run_train = False, run_test = True, model_path = None, checkpoint_freq = 10, train_params = None):
    if not os.path.exists(model_path + ".zip"):
        print(no_model_found_msg := f"Model checkpoint not found at {model_path}.zip. Training a new model.")
        run_train = True
    if model_path is None:
        print(no_model_path_msg := "No model path provided. Training a new model.")
        run_train = True

    if run_cluster:
        data_path = clustering.main() 
    else:
        data_path = "pointclouds/1_Denoise_NoVeg_Subsampled_centroid.npz"
    if run_train:
        latest_model = train.run_training(model_path=model_path, data_path=data_path, checkpoint_freq=checkpoint_freq, **train_params)
    else:
        latest_model = model_path
    if run_test:
        test.test_policy(checkpoint_freq=checkpoint_freq, model_path=latest_model, data_path=data_path)


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

    Test(run_cluster=RUN_CLUSTER, run_train=RUN_TRAIN, run_test=RUN_TEST, model_path=MODEL_PATH, checkpoint_freq=CHECKPOINT_FREQ, train_params=train_params)
