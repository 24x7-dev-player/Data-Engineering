import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Generating a sample 2D dataset
rng = np.random.RandomState(42)
X = 0.3 * rng.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))

# Combine the datasets
X = np.r_[X_train, X_outliers]

# Plotting the dataset before outlier detection
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], color='b', s=20, edgecolor='k')
plt.title("Original Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Fit the model
clf = IsolationForest(max_samples=100, random_state=rng, contamination='auto')
clf.fit(X)
y_pred = clf.predict(X)

clf.score_samples(X)

# Visualizing the data points and the outliers detected
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], color='b', s=20, edgecolor='k')

# Highlighting the outliers
is_outlier = y_pred == -1
plt.scatter(X[is_outlier, 0], X[is_outlier, 1], color='r', s=50, edgecolor='k')

plt.title("Outlier Detection with Isolation Forest")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

"""### KNN"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Generating more sample data
np.random.seed(0)
X_normal = np.random.randn(100, 2) * 2
X_outliers = np.array([[10, 10], [12, 12], [10, 14]])
X = np.concatenate([X_normal, X_outliers], axis=0)

# Plot the data before outlier detection
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], color='blue', label='Data Points')
plt.title('Data Points Before Outlier Detection')
plt.legend()
plt.show()

# Set the number of neighbors
k = 2

# Fit the model
nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
distances, indices = nbrs.kneighbors(X)

# Calculate the outlier score
outlier_scores = np.mean(distances[:, 1:], axis=1)

# Determine a threshold
threshold = np.percentile(outlier_scores, 90)  # using the 95th percentile as the threshold

# Identify outliers
outlier_indices = np.where(outlier_scores > threshold)[0]
outliers = X[outlier_indices]

# Plot the data after outlier detection
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], color='blue', label='Data Points')
plt.scatter(outliers[:, 0], outliers[:, 1], color='red', label='Outliers')
plt.title('Data Points After Outlier Detection')
plt.legend()
plt.show()

from sklearn.neighbors import NearestNeighbors

# Number of neighbors
k = 5

# Fit the model
nbrs = NearestNeighbors(n_neighbors=k + 1).fit(data)
distances, indices = nbrs.kneighbors(data)

# Compute the average distance to k nearest neighbors as the outlier score
outlier_scores = np.mean(distances[:, 1:], axis=1)

# Determine a threshold for outlier detection
# Here, we use the 95th percentile as the threshold
threshold = np.percentile(outlier_scores, 90)

# Points with a score above the threshold are considered outliers
is_outlier = outlier_scores > threshold

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(data[~is_outlier, 0], data[~is_outlier, 1], label='Normal Points')
plt.scatter(data[is_outlier, 0], data[is_outlier, 1], color='r', label='Outliers')
plt.title('Outlier Detection using kNN')
plt.legend()
plt.grid(True)
plt.show()

"""### Local Outlier Factor"""

from sklearn.neighbors import LocalOutlierFactor

# Applying LOF
clf_lof = LocalOutlierFactor(n_neighbors=10, contamination=0.05)
y_pred_lof = clf_lof.fit_predict(data)
lof_scores = -clf_lof.negative_outlier_factor_  # Inverting the negative LOF scores for better interpretation

# Identifying the outliers based on the LOF prediction
is_outlier_lof = y_pred_lof == -1

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(data[~is_outlier_lof, 0], data[~is_outlier_lof, 1], color='blue', label='Normal Points')
plt.scatter(data[is_outlier_lof, 0], data[is_outlier_lof, 1], color='red', label='Outliers')
plt.title('Outlier Detection using LOF')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from matplotlib.legend_handler import HandlerPathCollection

# Set seed for reproducibility
np.random.seed(0)

# Generate a dense cluster
dense_cluster = np.random.normal(loc=0, scale=0.5, size=(100, 2))

# Generate a sparse cluster
sparse_cluster = np.random.normal(loc=5, scale=2.5, size=(30, 2))

# Combine the clusters
X = np.concatenate([dense_cluster, sparse_cluster])

# Applying LOF
clf_lof = LocalOutlierFactor(n_neighbors=20)
y_pred_lof = clf_lof.fit_predict(X)
X_scores = -clf_lof.negative_outlier_factor_  # Negative scores, higher is more abnormal

# Calculate the radius of each circle, inversely proportional to the outlier score
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())

# Define a function to update legend marker size for clarity
def update_legend_marker_size(handle, orig):
    handle.update_from(orig)
    handle.set_sizes([20])

# Plotting the data points and circles representing outlier scores
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], color="k", s=3.0, label="Data points")
scatter = plt.scatter(
    X[:, 0], X[:, 1],
    s=1000 * radius,  # Adjust the multiplier for circle sizes as necessary
    edgecolors="r",
    facecolors="none",
    label="Outlier scores"
)
plt.axis("tight")
plt.xlim((X[:, 0].min() - 1, X[:, 0].max() + 1))
plt.ylim((X[:, 1].min() - 1, X[:, 1].max() + 1))
plt.legend(handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)},
           title="Legend")
plt.title("Local Outlier Factor (LOF) with Custom Data")
plt.show()

"""### DBSCAN"""

from sklearn.cluster import DBSCAN

# Applying DBSCAN to the dataset
dbscan = DBSCAN(eps=0.4, min_samples=7)  # These parameters can be adjusted
clusters = dbscan.fit_predict(data)

# Identifying the outliers
outliers = data[clusters == -1]

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], color='blue', label='Data Points', s=10)
plt.scatter(outliers[:, 0], outliers[:, 1], color='red', label='Outliers', s=10)
plt.title('Outlier Detection using DBSCAN')
plt.legend()
plt.grid(True)
plt.show()

"""### Comparison

| Feature/Technique      | Z-Score Technique                | IQR and Box Plot Method         | Isolation Forest                | KNN (k-Nearest Neighbors)      | LOF (Local Outlier Factor)     | DBSCAN (Density-Based Spatial Clustering of Applications with Noise) |
|------------------------|----------------------------------|---------------------------------|---------------------------------|--------------------------------|--------------------------------|---------------------------------------------------------------------|
| **Method Type**        | Statistical                     | Statistical                     | Ensemble Method                 | Distance-Based                 | Density-Based                  | Density-Based                                                       |
| **Approach**           | Standard deviation and mean     | Quartiles and median            | Random forest isolation         | Nearest neighbors distance     | Local density comparison       | Density-based clustering                                            |
| **Key Parameters**     | Threshold for Z-score (e.g., 3) | IQR multiplier (typically 1.5)  | Number of trees, path length    | Number of neighbors (k)        | Number of neighbors (k), radius | Epsilon (eps), MinPts                                              |
| **Sensitivity**        | Very sensitive to outliers      | Moderately sensitive            | Robust to outliers              | Sensitive to local outliers    | Sensitive to local density     | Sensitive to density variations                                     |
| **Scalability**        | Very scalable                   | Very scalable                   | Good scalability                | Poor with large datasets       | Poor with large datasets       | Poor with large datasets                                            |
| **Best Use Case**      | Gaussian-distributed data       | Non-Gaussian, not too skewed    | High-dimensional, mixed feature types | Small, low-dimensional datasets | Varying density clusters      | Spatial data with clear density gaps                                |
| **Interpretability**   | Very interpretable              | Highly interpretable            | Less interpretable              | Moderately interpretable       | Less interpretable             | Moderately interpretable                                            |
| **Advantages**         | Easy to understand and implement| Easy to understand, robust to mild outliers | Handles large feature sets well| Intuitive, effective in many scenarios | Effective in detecting outliers in varying density data | Good at identifying clusters and noise                              |
| **Disadvantages**      | Assumes normality, not good for multimodal data | Can miss outliers in a skewed distribution | Requires parameter tuning, complex | Sensitive to k, not good for high-dimensional data | Requires parameter tuning, complex | Parameters eps and MinPts can be hard to set, not ideal for high-dimensional data |
| **Output**             | Binary classification (inlier/outlier) | Binary classification (inlier/outlier) | Outlier score (continuous)     | Binary classification or score | Outlier score (continuous)     | Cluster labels (including noise)                                    |
| **Model Type**         | Unsupervised                    | Unsupervised                    | Unsupervised                    | Unsupervised or semi-supervised| Unsupervised                    | Unsupervised                                                        |
| **Data Assumption**    | Assumes feature independence    | Assumes feature independence    | No assumption on data structure | Assumes local similarity       | Assumes local similarity       | Assumes clusters are dense regions of points                        |
| **Robustness**         | Not robust to skewed data       | More robust than Z-score        | Robust to isolated noise        | Moderate robustness            | Robust to local density changes| Robust to cluster shape variations                                  |
"""

