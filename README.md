# AIIR-TheCoolKindOfAI

/pointclouds contains raw .laz LiDAR scans from Spatial NSW.
    - Each sub folder contains the metadata pertaining to the original scan, the original scan, and a file named "denoised" that has had all scalar layers stripped (using CloudCompare FOSS) except for layer 2 (Ground), 5 (Vegetation, disregarded when loaded), and 6 (Buildings). Denoised files may also be cropped to a smaller area.

/src contains the code for each step of the process
    - clustering.py -> Reads in a specified Denoised LAZ file from the pointclouds folder and trains a clasifier to identify ground vs buildings. Then it takes the output from the classifier and constructs box shaped obstacles whose size and position is passed to environment.py
    - environment.py -> Constructs and sets up the gymnasium environment
    - train.py -> Trains the car to drive to the goal
    - test.py -> Tests trained car AI model
    - main.py -> Runs the main sequence of events, (custering -> environment -> train -> test)
