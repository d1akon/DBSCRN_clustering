from dbscrn.algorithm import DBSCRN
from dbscrn.visualizer import Visualizer
from utils.utils import analyze_dataset
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from dbscrn.dunn_index import dunn
from dbscrn.gap_statistic import gap_statistic


def main():
    from sklearn.datasets import make_moons, make_circles, make_blobs, make_classification

    datasets = [
        ("Moons", lambda: make_moons(n_samples=1000, noise=0.05)),
        ("Circles", lambda: make_circles(n_samples=1000, factor=0.5, noise=0.05)),
        ("Blobs", lambda: make_blobs(n_samples=1000, centers=4, cluster_std=1.0, random_state=42)),
        ("Classification", lambda: make_classification(n_samples=1000, n_features=2, n_clusters_per_class=1, n_classes=4, n_redundant=0, random_state=42)),
    ]

    for dataset_name, dataset_func in datasets:
        print(f"\n=== Processing Dataset: {dataset_name} ===")
        data, _ = dataset_func()

        # Analyze the dataset to determine the type of problem
        problem_type = analyze_dataset(data)
        print(f"Estimated problem type: {problem_type}")

        # Set metrics based on problem type
        if problem_type == "density":
            metrics = [davies_bouldin_score, dunn]
        elif problem_type == "classic":
            metrics = [silhouette_score, calinski_harabasz_score, gap_statistic]
        else:  # Mixed or undefined
            metrics = [davies_bouldin_score, calinski_harabasz_score]

        # Initialize algorithm
        algorithm = DBSCRN(k=15)

        # Find the optimal k based on selected metrics
        k_values = range(5, 51, 5)
        best_k, best_clusters, labels = algorithm.find_optimal_k(data, k_values, metrics)

        print(f"Best k for {dataset_name}: {best_k}")
        Visualizer.plot_clusters(data, best_clusters)



if __name__ == "__main__":
    main()
