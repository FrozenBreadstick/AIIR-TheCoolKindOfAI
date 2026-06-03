# API Reference

<hr style="border: 2px solid #0071ea;">

## clustering.py

??? info "Click to expand this section"

    ### main {: #api_cluster_main }
    ::: clustering.main

    ---

    ### load_laz {: #api_load }
    ::: clustering.load_laz

    ---

    ### visualize {: #api_visualise }
    ::: clustering.visualize

    ---

    ### DavidBentleyScan {: #api_dbscan }
    ::: clustering.DavidBentleyScan

    ---

    ### FelicityRandomForest {: #api_forest }
    ::: clustering.FelicityRandomForest

    ---

    ### CericCentroid {: #api_centroid }
    ::: clustering.CedricCentroid

<hr style="border: 2px solid #0071ea;">
<hr style="border: 2px solid #ff53fc90;">

## train.py
 
??? info "Click to expand this section"
 
    ### custom_observation {: #api_custom_observation }
 
    ::: train.custom_observation
 
    !!! note
        If `checkpoint_pos` is `None`, the goal position is used as a dummy checkpoint value so the observation shape stays consistent.
 
    ---
 
    ### custom_reward {: #api_custom_reward }
 
    ::: train.custom_reward
 
    **Reward breakdown:**
 
    | Component | Value |
    | :--- | :--- |
    | Step penalty | `-0.2` per step |
    | Goal reached | `+200.0` |
    | Checkpoint reached | `+150.0` |
    | Progress reward (goal) | `+10.0 × (prev_dist - cur_dist)` |
    | Progress reward (checkpoint) | `+10.0 × (prev_dist - cur_dist)` |
    | LiDAR proximity penalty | `-3.0 × (0.03 - min_lidar)` when `min_lidar < 0.03` |
    | LiDAR danger penalty | `-12.0 × (0.02 - min_lidar)` when `min_lidar < 0.02` |
    | Collision penalty | `-400.0` |
 
    ---
 
    ### run_training {: #api_run_training }
 
    ::: train.run_training
 
<hr style="border: 2px solid #0071ea;">
<hr style="border: 2px solid #ff53fc90;">

## test.py
 
??? info "Click to expand this section"
 
    ### test_policy {: #api_test_policy }
 
    ::: test.test_policy
 
    **Scenarios run:**
 
    The model is tested in 3 separate scenarios each with a randomly generated environment. Each time the environment is reset, the policy runs deterministically (`deterministic=True`) until `terminated` or `truncated` is set, and the total episode reward is printed to stdout.
 
<hr style="border: 2px solid #ff53fc90;">

