import argparse
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import open3d as o3d
import polyscope as ps
import laspy

from sklearn.cluster import DBSCAN #We will use DBSCAN for clustering
from sklearn.svm import SVC

# Additionally, you may want to use StandardScaler for height normalisation before clustering
from sklearn.preprocessing import StandardScaler

# Libraries for Profiling and testing
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report

# Will need to Downsample the ply model using Open3D
# Then run SVC or RandomForestClassifier on the data to seperate floor from obstacles (Classify on a per point basis, this is why we downsample)
# Run DBSCAN to group obstacles points together and find ceontroids and identify bounding boxes for environment creation

# We are using Laz cus it is more storage space efficient. Laz v1.4 (Point Format 0)
def load_laz(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    lazarus = laspy.read(path)
    clarissa = lazarus.classification
    print(np.unique(clarissa))
    points = None
    return points, clarissa


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def visualize(points: np.ndarray, labels: np.ndarray, k: int) -> None:
    """
    Set up Polyscope serialization.
    """
    print("Visualizing results in Polyscope...")
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")

    # 1. Register main point cloud
    cloud = ps.register_point_cloud("Processed Point Cloud", points, radius=0.0015)
    cloud.set_point_render_mode("quad")
    cloud.add_scalar_quantity("Elevation", points[:, 2], enabled=False)

    # 2. Add ground truth labels
    cloud.add_scalar_quantity("Ground truth data", labels, enabled=False)
    
    # 3. Add colors for clusters
    cluster_labels = np.load("cluster_labels.npy")
    cloud.add_scalar_quantity("Clusters", cluster_labels, enabled=True)

    # 4. Register PCA visualization (Cluster Centers and Principal Components)
    # you can register pointclouds: https://polyscope.run/py/structures/point_cloud/basics/
    # you can then add vector quantities: https://polyscope.run/py/structures/point_cloud/vector_quantities/
    data = np.load("features.npz")
    centers = data["center"]
    pc1 = data["pc1"]
    pc2 = data["pc2"]
    pc3 = data["pc3"]
    variance = data["variance"]
    pca_cloud = ps.register_point_cloud("Cluster PCA Centers", centers, radius=0.005)
    #Scale by eigenvalues for magnitude visualisation + arbitrary scale value to keep them within reasonable range
    pca_cloud.add_vector_quantity("PC1 (Major)", pc1 * variance[:, 0:1] * 0.05, enabled=True, color=(1, 0, 0), vectortype="ambient")
    pca_cloud.add_vector_quantity("PC2 (Minor)", pc2 * variance[:, 1:2] * 0.05, enabled=True, color=(0, 1, 0), vectortype="ambient")
    pca_cloud.add_vector_quantity("PC3 (Normal)", pc3 * variance[:, 2:3] * 5, enabled=True, color=(0, 0, 1), vectortype="ambient")

    # 4.5. Per Cluster Ground Truth Visualisation
    gts = np.load("per_cluster_gt.npy")
    point_cluster_gt_labels = gts[cluster_labels]
    cloud.add_scalar_quantity("Per Cluster GT Majority", point_cluster_gt_labels, enabled=False)

    # 5. Add SVM Predictions
    pred = np.load("predictions.npy")
    cloud.add_scalar_quantity("SVM Predictions", pred, enabled=False)
    ps.show()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Point Cloud Segmentation pipeline (K-Means -> PCA -> SVM)")
    parser.add_argument("path", nargs="?", default="pointclouds/1/Denoise_NoVeg.laz")
    parser.add_argument("-k", "--clusters", type=int, default=15, help="Number of k-means clusters (default: 20)")
    args = parser.parse_args()


    # Load Data
    points, point_gt_labels = load_laz(args.path)
    print(f"Loaded {len(points)} points from {args.path}")
    return

    # Downsampling
    downsample = downsample_points()
    np.save("downsampled_points.npy", downsample)

    # Classification
    ### svm after make features

    center, pc1, pc2, pc3, variance = pca(points, k_labels, opt_k)
    np.savez(
        "features.npz",
        center=center,
        pc1=pc1,
        pc2=pc2,
        pc3=pc3,
        variance=variance
    )

    # Clustering
    #### DBSCAN to find obstacles, filter out the points that are part of the floor

    # Centroid & Non-AA BB Identification

    # OLD CODE
    # 4. Ground Truth Generation (for training SVM)
    gts = truth(opt_k, point_gt_labels, k_labels)
    np.save("per_cluster_gt.npy", gts)
    #Construct feature list and hstack to obtain feature matrix
    X = make_feature(center, pc1, pc2, pc3)

    # 5. SVM Classification
    predictions = svm(gts, X, k_labels)
    np.save("predictions.npy", predictions)

    # 6. Visualization
    visualize(points, point_gt_labels, args.clusters)

def k_means(points: np.ndarray, k: int = 20) -> tuple[np.ndarray, int]:
    print(f"Running k-means with k = " + str(k))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(points)
    
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(scaled)
    
    return kmeans.labels_, k

def k_means_optimal(points: np.ndarray, k_max: int = 15) -> tuple[np.ndarray, int]:
    #Scale data
    scaler = StandardScaler()
    scaled = scaler.fit_transform(points)
    clusters = []
    inertias = []
    #Test for all k values up to k_max
    for k in range(1, k_max + 1):
        print("Testing k = " + str(k))
        kmeans = KMeans(n_clusters = k)
        kmeans.fit(scaled)
        clusters.append(kmeans.labels_)
        inertias.append(kmeans.inertia_)
    
    last_best = 0
    best_index = 0
    print("Finding optimal k")
    #Find optimal k using elbow method, find greatest change in gradient
    for i in range(1, len(inertias) - 1):
        prev_grad = inertias[i - 1] - inertias[i]
        next_grad = inertias[i] - inertias[i + 1]
        change = prev_grad - next_grad
        if change > last_best:
            last_best = change
            best_index = i
    print("Optimal k = " + str(best_index+1))

    #Return optimal K and the labels belonging to that K
    return np.asarray(clusters[best_index]), best_index+1

def pca(points: np.ndarray, k_labels: np.ndarray, k: int):
    #Lists to store components
    center = []
    pc1 = []
    pc2 = []
    pc3 = []
    variance = []
    #Run PCA on each cluster
    print("Running PCA Feature Extraction")
    for i in range(k):
        #Find clusters 
        cluster_points = points[k_labels == i]
        pca = PCA(n_components=3)
        pca.fit(cluster_points)
        center.append(pca.mean_)
        pc1.append(pca.components_[0])
        pc2.append(pca.components_[1])
        pc3.append(pca.components_[2])
        variance.append(pca.explained_variance_)
    return np.asarray(center), np.asarray(pc1), np.asarray(pc2), np.asarray(pc3), np.asarray(variance)

def truth(opt_k: int, gt_labels: np.ndarray, k_labels: np.ndarray) -> np.ndarray:
    gt = []
    for i in range(opt_k):
        cluster_labels = gt_labels[k_labels == i] #Get a list of every ground truth label in that cluster
        gt.append(np.bincount(cluster_labels).argmax()) #Use the most common ground truth as the ground truth for that cluster
    return np.asarray(gt)

def make_feature(center, pc1, pc2, pc3):
    features_list = [center, pc1, pc2, pc3]
    x = np.hstack(features_list)
    return x

def svm(gt: np.ndarray, X: np.ndarray, k_labels: np.ndarray):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #Tuned Values, Linear SVC Model, Balanced Class weights to ensure each class has an equal chance
    svc = SVC(kernel='linear', C=10, gamma=0.01, class_weight='balanced')
    print("Training SVM in Cross Validation")
    #Run Cross validation predictions, split to 10 parts and train on 90% of data
    cv_predictions = cross_val_predict(svc, X_scaled, gt, cv=10)
    print("Cross-Validation Report:")
    #Print the classification report
    print(classification_report(gt, cv_predictions))

    svc.fit(X_scaled, gt)
    prediction = svc.predict(X_scaled)
    return prediction[k_labels]

if __name__ == "__main__":
    main()