import warnings

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.preprocessing import MinMaxScaler


class ClusterAnomalyDetector(BaseEstimator, OutlierMixin):
    def __init__(self, clusterer, contamination: float = 0.1, scale_scores: bool = True):
        """
        Anomaly detector based on clustering distance.

        This class implements an anomaly detection model using a given clustering algorithm.
        The distance of a sample to its assigned cluster centroid is used as an anomaly score.
        Higher scores indicate potential anomalies.

        Parameters
        ----------
        clusterer : object
            A scikit-learn-compatible clustering model (e.g., KMeans, Birch, etc.).
            The model must implement a `fit` method and either a `predict` or `fit_predict` method.

        contamination : float, default=0.1
            The proportion of anomalies in the dataset, which determines the threshold for
            anomaly classification. Must be in the range (0, 0.5].

        scale_scores : bool, default=True
            If True, distances are scaled using Min-Max scaling for better interpretability.

        Attributes
        ----------
        centroids_ : ndarray of shape (n_clusters, n_features) or None
            The computed centroids of the clusters. If the clustering model does not expose
            centroids, they are estimated using the mean of assigned cluster points.

        threshold_ : float
            The threshold above which a sample is classified as an anomaly.

        labels_ : ndarray of shape (n_samples,)
            The cluster labels assigned to each sample during fitting.

        scaler_ : object
            MinMaxScaler instance used to normalize anomaly scores if `scale_scores=True`.
        """

        if not (hasattr(clusterer, "fit") and (hasattr(clusterer, "predict") or hasattr(clusterer, "fit_predict"))):
            raise ValueError("The provided model does not seem to be a scikit-learn or compatible clustering model. "
                             "The model needs to contain  'fit', 'predict' or 'fit_predict' methods.")
        else:
            if hasattr(clusterer, "fit_predict") and not hasattr(clusterer, "predict"):
                warnings.warn("Clusterers with no predict require double fitting in both fit and score_samples/predict.")

            self.clusterer = clusterer
            self.contamination = contamination
            self.scale_scores = scale_scores
            self.scaler_ = MinMaxScaler()
            self.centroids_ = None
            self.threshold_ = None
            self.labels_ = None

    def fit(self, X):
        """
        Fit the clustering model and compute anomaly detection threshold.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data for clustering and anomaly detection.

        Returns
        -------
        self : object
            Returns the fitted instance of `ClusterAnomalyDetector`.
        """

        if hasattr(self.clusterer, "fit"):
            self.clusterer.fit(X)
        else:
            self.labels_ = self.clusterer.fit_predict(X)

        if hasattr(self.clusterer, "cluster_centers_"):
            self.centroids_ = self.clusterer.cluster_centers_
        elif hasattr(self.clusterer, "means_"):
            self.centroids_ = self.clusterer.means_
        elif hasattr(self.clusterer, "subcluster_centers_"):
            self.centroids_ = self.clusterer.subcluster_centers_
        elif hasattr(self.clusterer, "labels_"):
            warnings.warn("Clusterers which return anomaly labels (e.g. DBSCAN) will cause a discrepancy between fit"
                          " and score_samples where in score samples the anomaly score for already anomalous samples"
                          " will be np.inf or self.scaler_.data_max_ based on self.scale_scores. To suppress the "
                          "warning switch over to a non-anomaly detection clusterer.")
            labels = self.clusterer.labels_
            unique_labels = np.setdiff1d(np.unique(labels), -1)
            centroids = np.array([X[np.where(labels == label)].mean(axis=0) for label in unique_labels])
            self.centroids_ = centroids
        else:
            raise ValueError("The clusterer does not provide any centroids or class labels.")

        percentile_point = 100 - (self.contamination * 100)

        if hasattr(self.clusterer, "predict"):
            cluster_labels = self.clusterer.predict(X)
        elif hasattr(self.clusterer, "fit_predict"):
            cluster_labels = self.clusterer.fit_predict(X)
        else:
            raise ValueError("The provided clusterer does not support prediction.")

        distances = np.array([np.linalg.norm(self.centroids_[label] - sample) for label, sample in zip(cluster_labels, X)])
        if self.scale_scores:
            self.scaler_.fit(distances.reshape(-1, 1))
            distances = self.scaler_.transform(distances.reshape(-1,1))
            self.threshold_ = np.percentile(distances, percentile_point)
        else:
            self.threshold_ = np.percentile(distances, percentile_point)

        return self

    def score_samples(self, X):
        """
        Compute anomaly scores for each sample based on its distance to the nearest cluster centroid.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Anomaly scores, where higher values indicate greater likelihood of being an anomaly.
        """
        if hasattr(self.clusterer, "predict"):
            cluster_labels = self.clusterer.predict(X)
        elif hasattr(self.clusterer, "fit_predict"):
            cluster_labels = self.clusterer.fit_predict(X)
        else:
            raise ValueError("The provided clusterer does not support prediction.")

        distances = list()
        for label, sample in zip(cluster_labels, X):
            if label != -1:
                centroid = self.centroids_[label]
                distance = np.linalg.norm(centroid - sample)
                distances.append(distance)
            else:
                if self.scale_scores:
                    distances.append(self.scaler_.data_max_[0])
                else:
                    distances.append(np.inf)

        distances = np.array(distances)
        if self.scale_scores:
            distances = self.scaler_.transform(distances.reshape(-1, 1)).flatten()
        return distances

    def predict(self, X):
        """
        Predict whether each sample is an anomaly or not.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Binary classification labels: -1 for anomalies and 1 for normal instances.
        """

        scores = self.score_samples(X)
        y = np.where(scores >= self.threshold_, -1, 1)
        return y
