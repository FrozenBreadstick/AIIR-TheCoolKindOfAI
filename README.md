# AIIR-TheCoolKindOfAI

/pointclouds contains raw .laz LiDAR scans from Spatial NSW.
    - Each sub folder contains the metadata pertaining to the original scan, the original scan, and a file named "denoised_npveg_subsampled" which removes random lidar noise included in the file, strips all points except for buildings and ground, and subsamples the points to 10% of the original model (~3mil points -> 300k)

/src contains the code for each step of the process
    - clustering.py -> Reads in a specified Denoised LAZ file from the pointclouds folder and trains a clasifier to identify ground vs buildings. Then it takes the output from the classifier and constructs box shaped obstacles whose size and position is passed to environment.py
    - environment.py -> Constructs and sets up the gymnasium environment
    - train.py -> Trains the car to drive to the goal
    - test.py -> Tests trained car AI model
    - main.py -> Runs the main sequence of events, (custering -> environment -> train -> test)

/tensor_data contains previous trained models and figures from the tensorboard at time of training finish
    - ppo39.10 refers to a model that was trained with goals every 10 metres
    - ppo39.30 refers to a model that was trained with goals every 30 metres
    - ppo39.30_2 refers to a more extensively trained model with goals every 30 metres
    - ppo39.40_137 refers to a model that was trained with goals every 40 metres and had 13 700 000 steps of training.

    note: that these models are trained using the previous model as a starting point. ppo39.40_137 started training at 7 700 000 steps where ppo39.30_2 left off.
