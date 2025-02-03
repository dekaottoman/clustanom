import warnings

from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, KMeans, MiniBatchKMeans, MeanShift, OPTICS, SpectralClustering, FeatureAgglomeration, SpectralBiclustering, SpectralCoclustering
from sklearn.cluster import estimate_bandwidth
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.datasets import make_blobs

from clustanom.detector import ClusterAnomalyDetector

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    X, _ = make_blobs(n_samples=1000, centers=3, random_state=42)

    model_dict = {
        "AffinityPropagation": AffinityPropagation(damping=0.9, max_iter=200, convergence_iter=15, random_state=42),
        "AgglomerativeClustering": AgglomerativeClustering(n_clusters=3, linkage='ward'),
        "Birch": Birch(threshold=0.5, branching_factor=50, n_clusters=3),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5, metric='euclidean'),
        "KMeans": KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, random_state=42),
        "MiniBatchKMeans": MiniBatchKMeans(n_clusters=3, init='k-means++', max_iter=100, batch_size=100, random_state=42),
        "MeanShift": MeanShift(bandwidth=estimate_bandwidth(X, quantile=0.2, n_samples=500), bin_seeding=True),
        "OPTICS": OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.1),
        "SpectralClustering": SpectralClustering(n_clusters=3, affinity='nearest_neighbors', assign_labels='kmeans', random_state=42),
        "FeatureAgglomeration": FeatureAgglomeration(n_clusters=3),
        "SpectralBiclustering": SpectralBiclustering(n_clusters=(3, 3), method='bistochastic', random_state=0),
        "SpectralCoclustering": SpectralCoclustering(n_clusters=3, random_state=0),
        "GaussianMixture": GaussianMixture(n_components=3, covariance_type='full', random_state=42),
        "BayesianGaussianMixture": BayesianGaussianMixture(n_components=10, covariance_type='full', random_state=42)
    }

    for key, model in model_dict.items():
        print(f"Testing for clusterer {key}")

        definition_pass = True
        fit_pass = True
        score_samples_pass = True
        predict_pass = True

        try:
            anomaly_detector = ClusterAnomalyDetector(model)
            print(f"{'-'*4}Passed model definition test for clusterer {key}")

            try:
                anomaly_detector.fit(X)
                print(f"{'-'*4}Passed fit test for clusterer {key}")
            except Exception as e:
                print(f"{'!'*4}Error fitting model with clusterer {key} >> {e}")

            try:
                anomaly_detector.score_samples(X)
                print(f"{'-'*4}Passed score_samples test for clusterer {key}")
            except Exception as e:
                print(f"{'!'*4}Error scoring samples with clusterer {key} >> {e}")

            try:
                anomaly_detector.predict(X)
                print(f"{'-'*4}Passed predict test for clusterer {key}")
            except Exception as e:
                print(f"{'!'*4}Error predicting with clusterer {key} >> {e}")

        except Exception as e:
            print(f"{'!'*4}Error creating model with clusterer {key} >> {e}")

        print("\n")
