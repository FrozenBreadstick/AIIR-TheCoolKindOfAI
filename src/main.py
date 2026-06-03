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
    test.test_policy()

if __name__ == "__main__":
    Test()
