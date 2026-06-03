#Imports
import os
import argparse

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

LAZ_PATH = "pointclouds/1/Denoise_NoVeg_Subsampled.laz"

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

def Test(run_cluster = True, run_train = False, run_test = True, model_path = None, checkpoint_freq = 10, train_params = None, laz = None):
    if not os.path.exists(model_path + ".zip"):
        print(no_model_found_msg := f"Model checkpoint not found at {model_path}.zip. Training a new model.")
        run_train = True
    if model_path is None:
        print(no_model_path_msg := "No model path provided. Training a new model.")
        run_train = True

    if run_cluster:
        if laz is None:
            data_path = clustering.main() 
        else:
            data_path = clustering.main(laz)
    else:
        data_path = "pointclouds/1_Denoise_NoVeg_Subsampled_centroid.npz"
    if run_train:
        latest_model = train.run_training(model_path=model_path, data_path=data_path, checkpoint_freq=checkpoint_freq, **train_params)
    else:
        latest_model = model_path
    if run_test:
        test.test_policy(checkpoint_freq=checkpoint_freq, model_path=latest_model, data_path=data_path)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration for clustering, training, and testing.")
    
    # General Configuration Args
    parser.add_argument("--run_cluster", type=str2bool, default=RUN_CLUSTER, help="Whether to run the clustering script")
    parser.add_argument("--run_train", type=str2bool, default=RUN_TRAIN, help="Whether to run the training process")
    parser.add_argument("--run_test", type=str2bool, default=RUN_TEST, help="Whether to run the testing process")
    parser.add_argument("--checkpoint_freq", type=int, default=CHECKPOINT_FREQ, help="Checkpoint frequency")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to the model")
    parser.add_argument("--laz", type=str, default=LAZ_PATH, help="Path to the LAZ file")

    # Training Configuration Args
    parser.add_argument("--total_timesteps", type=int, default=TOTAL_TIMESTEPS, help="Total timesteps for training")
    parser.add_argument("--n_envs", type=int, default=N_ENVS, help="Number of environments")
    parser.add_argument("--n_steps", type=int, default=N_STEPS, help="Number of steps")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--n_epochs", type=int, default=N_EPOCHS, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--entropy_coef", type=float, default=ENTROPY_COEF, help="Entropy coefficient")
    parser.add_argument("--gae_lambda", type=float, default=GAE_LAMBDA, help="GAE Lambda")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="Discount factor (Gamma)")
    parser.add_argument("--max_grad_norm", type=float, default=MAX_GRAD_NORM, help="Maximum gradient norm")
    parser.add_argument("--clip_range", type=float, default=CLIP_RANGE, help="Clip range")

    args = parser.parse_args()

    train_params = {
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "learning_rate": args.learning_rate,
        "entropy_coef": args.entropy_coef,
        "gae_lambda": args.gae_lambda,
        "gamma": args.gamma,
        "max_grad_norm": args.max_grad_norm,
        "clip_range": args.clip_range
    }

    Test(
        run_cluster=args.run_cluster, 
        run_train=args.run_train, 
        run_test=args.run_test, 
        model_path=args.model_path, 
        checkpoint_freq=args.checkpoint_freq, 
        train_params=train_params, 
        laz=args.laz
    )
