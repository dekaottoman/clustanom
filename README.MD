# ClustAnom - Clustering Based Anomaly Detection

ClustAnom is a scikit-learn compatible anomaly detection framework that utilizes user-defined clustering algorithms. It determines anomalies based on the distance of data points from their assigned cluster centers. This flexible approach allows integration with various clustering methods to detect outliers effectively.

## 🔍 Features

- Supports user-defined clustering models.
- Measures anomaly scores based on distance from cluster centroids.
- Fully compatible with scikit-learn pipelines.
- Easy integration with existing machine learning workflows.

## 🚀 Supported Clustering Algorithms
ClustAnom supports clustering models that implement `fit`, `predict` or `fit_predict` methods:

- `AffinityPropagation`
- `AgglomerativeClustering`
- `Birch`
- `DBSCAN`
- `KMeans`
- `MiniBatchKMeans`
- `MeanShift`
- `OPTICS`
- `SpectralClustering`
- `GaussianMixture`
- `BayesianGaussianMixture`

### ❌ Unsupported Clustering Algorithms
Explicitly unsupported clustering algorithms include, but are not limited to:

- `FeatureAgglomeration`
- `SpectralBiclustering`
- `SpectralCoclustering`

### ⚠️ Warnings
Clusterers which return anomaly labels (e.g. `DBSCAN`) will cause a discrepancy between `fit` and `score_samples` where in `score_samples` the anomaly score for already anomalous samples will be `np.inf` or `self.scaler_.data_max_` based on `self.scale_scores`. Additionally, clusterers with no `predict` require double fitting in both `fit` and `score_samples` or `predict` which may cause additional discrepancies.

## 📦 Installation
You can install ClustAnom via PyPI:

```bash
pip install clustanom
```

## 🛠️ Usage Example

```python
from clustanom.detector import ClusterAnomalyDetector
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Sample data
X, _ = make_blobs(n_samples=1000, centers=3, random_state=42)

# Define clusterer
clusterer = KMeans(n_clusters=3, random_state=42)

# Initialize cluster anomaly detector
anomaly_detector = ClusterAnomalyDetector(clusterer)

# Fit, predict and score
anomaly_detector.fit(X)
predictions = anomaly_detector.predict(X)
anomaly_scores = anomaly_detector.score_samples(X)
print("Anomaly Scores:", anomaly_scores)
```

## ⚙️ Parameters 

- **`clusterer`**: Any scikit-learn compatible clustering model (e.g., KMeans, Birch, Gaussian Mixture Model).
- **`contamination`**: The proportion of anomalies in the dataset, which determines the threshold for anomaly classification.
- **`scale_scores`**: Whether to scale the distances using MinMax scaling for better interpretability (limits anomaly scores to `[0, 1]`).


## ⚙️ Methods

- **`fit(X)`**: Fits the clustering model and computes the anomaly detection threshold.
- **`score_samples(X)`**: Computes anomaly scores for each sample based on its distance to the nearest cluster centroid.
- **`predict(X)`**: Predicts whether each sample is an anomaly (`-1`) or normal (`1`) in accordance with scikit-learn conventions.

## 📜 License

This project is licensed under the MIT License.
