import argparse
from typing import Optional, Tuple, List, Dict, Any
import os
import time

import numpy as np
import polyscope as ps
import laspy

import hdbscan #for db scan, allows streaming of points so we dont run out of memory
from sklearn.ensemble import RandomForestClassifier

# We are using Laz cus it is more storage space efficient. Laz v1.4 (Point Format 0)
def load_laz(path: str, filename: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    print("Loading Points")
    percicus = None
    clarissa = None
    if os.path.exists(filename + "_ground_truth.npz"):
        print("Existing Save Data Exists, Loading...")
        data = np.load(filename + "_ground_truth.npz")
        percicus = data["points"]
        clarissa = data["gt"]
    else:
        print("No Existing Save")
        lazarus = laspy.read(path)
        stanley = {2: 0, 6: 1}
        clarissa = np.vectorize(stanley.get)(lazarus.classification)
        print(np.unique(clarissa))
        percicus = np.vstack((lazarus.x, lazarus.y, lazarus.z)).T
        np.savez(
            filename + "_ground_truth.npz",
            points = percicus,
            gt = clarissa
        )
    print(f"Loaded {len(percicus)} points from {path}")
    return percicus, clarissa

def visualize(filename: str) -> None:
    """
    Set up Polyscope serialization.
    """
    print("Visualizing results in Polyscope...")
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")

    point_data = np.load(filename + "_ground_truth.npz")
    gt_points = point_data["points"]
    gt_labels = point_data["gt"]

    # Point cloud
    cloud = ps.register_point_cloud("City Point Cloud", gt_points, radius=0.001)
    cloud.set_point_render_mode("quad")
    cloud.add_scalar_quantity("Elevation", gt_points[:, 2], enabled=False)

    # Ground truth
    cloud.add_scalar_quantity("Raw GT", gt_labels, enabled=True)

    # Classifier Predictions
    prediction_labels = np.load(filename + "_classified.npy")
    cloud.add_scalar_quantity("Predictions", prediction_labels, enabled=False)

    # Cluster Labels
    cluster_labels = np.load(filename + "_cluster_labels.npy")
    cloud.add_scalar_quantity("Clusters", cluster_labels, enabled=False)

    #Centroids & Bounds
    centroid_data = np.load(filename + "_centroid.npz")
    centroids = centroid_data["centroids"]
    metrics = centroid_data["metrics"]
    centroids3 = np.hstack([centroids, np.full((len(centroids), 1), 90.0)])
    centers = ps.register_point_cloud("Cluster Centers", centroids3, radius=0.005)
    all_nodes = []
    all_edges = []
    offset = 0

    #connect each set of points
    edges = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0]
    ])

    for m in metrics:
        #m is stored as [north, east, south, west]
        quad = np.array(m)
        quad3 = np.hstack([quad, np.full((4, 1), 90.0)])
        all_nodes.append(quad3)
        all_edges.append(edges + offset)
        offset += 4
    all_nodes = np.vstack(all_nodes)
    all_edges = np.vstack(all_edges)

    ps.register_curve_network("Building Boundaries", all_nodes, all_edges, radius=0.0015, color=[255,0,0])
    ps.show()

# DBSCAN clusterer
def DavidBentleyScan(points: np.ndarray, gts: np.ndarray, filename: str) -> np.ndarray:
    print("Starting Clustering")
    print("Stripping Ground for XY Clustering")
    start = time.time()
    labels = -1 * np.ones(len(points), dtype=int)
    b_points = points[gts == 1]
    if os.path.exists(filename + "_cluster_labels.npy"):
        print("Existing Save Data Exists, Loading...")
        labels = np.load(filename + "_cluster_labels.npy")
    else:
        print("No Existing Save")
        henry = hdbscan.HDBSCAN(
            min_cluster_size=50,
            min_samples=60,
            core_dist_n_jobs=-1
        )
        cluster_labels = henry.fit_predict(b_points[:, :2])
        labels[gts == 1] = cluster_labels
        np.save(filename + "_cluster_labels.npy", labels)
    end = time.time()
    print("Clustering Complete in:", end - start, "seconds")
    return labels

# Need to switch to training on 80% testing on 20% or some other split
def FelicityRandomForest(points: np.ndarray, gts: np.ndarray, filename: str) -> np.ndarray:
    print("Starting Forest Classification")
    felicity = None
    start = time.time()
    if os.path.exists(filename + "_classified.npy"):
        print("Existing Save Data Exists, Loading...")
        felicity = np.load(filename + "_classified.npy")
    else:
        print("No Existing Save")
        felix = RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1
        )
        felix.fit(points, gts)
        felicity = felix.predict(points)
        np.save(filename + "_classified.npy", felicity)
    end = time.time()
    print("Classifying Complete in: " + str(end-start) + " seconds")
    return felicity

def CedricCentroid(points: np.ndarray, cluster_labels: np.ndarray, filename: str) -> Tuple[np.ndarray, np.ndarray]:
    print("Starting Centroid & Bounds Calculations")
    centroids = []
    metrics = []
    if os.path.exists(filename + "_centroid.npz"):
        print("Existing Save Data Exists, Loading...")
        cedric = np.load(filename + "_centroid.npz")
        centroids = cedric["centroids"]
        metrics = cedric["metrics"]
    else:
        print("No Existing Save")

        for i in np.unique(cluster_labels):
            if i == -1:
                continue
            cluster_points = points[cluster_labels == i]
            centroid = np.mean(cluster_points[:, :2], axis=0)
            pts = cluster_points[:, :2]

            #cardinal extreme points, general shape representing building
            north = cluster_points[np.argmax(pts[:, 1])]
            south = cluster_points[np.argmin(pts[:, 1])]
            east  = cluster_points[np.argmax(pts[:, 0])]
            west  = cluster_points[np.argmin(pts[:, 0])]

            centroids.append(centroid)
            metrics.append(np.array([north[:2], east[:2], south[:2], west[:2]]))

        np.savez(
            filename + "_centroid.npz",
            centroids=np.array(centroids),
            metrics=np.array(metrics)
        )

    print("Centroids and Bounds Found")
    return np.array(centroids), np.array(metrics)

def main() -> None:
    parser = argparse.ArgumentParser(description="Point Cloud Segmentation pipeline (K-Means -> PCA -> SVM)")
    parser.add_argument("path", nargs="?", default="pointclouds/1/Denoise_NoVeg_Subsampled.laz")
    parser.add_argument("-k", "--clusters", type=int, default=15, help="Number of k-means clusters (default: 20)")
    args = parser.parse_args()

    # Load Data
    filename = args.path.rsplit('/', 1)[-1]
    filename = filename.replace(".laz", "") # Fix filepath
    points, original_gt_labels = load_laz(args.path, filename)

    # Classify with RandomForest on a point by point basis
    predict_gt_labels = FelicityRandomForest(points, original_gt_labels, filename)

    # DBSCAN Cluster
    cluster_labels = DavidBentleyScan(points, predict_gt_labels, filename)

    #Find Centroids and Bounds
    CedricCentroid(points, cluster_labels, filename)
    visualize(filename)
    return

if __name__ == "__main__":
    main()
