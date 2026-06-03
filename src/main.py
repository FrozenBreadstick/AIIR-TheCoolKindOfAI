#Imports
import sys
import os
import time
import logging

import clustering
import train
import test

def Test():
    clustering.main()
    train.run_training()
    test.test_policy(checkpoint_freq=40, model_path="model\checkpoints\ppo_driving_13700000_steps.zip")

if __name__ == "__main__":
    Test()
