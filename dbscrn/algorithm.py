import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from dbscrn.dunn_index import dunn  # Import your custom Dunn index implementation
from typing import List, Dict

class Point:
    """Represents a point in the dataset."""
    def __init__(self, coordinates: np.ndarray):
        self.coordinates = coordinates
        self.cluster = None  # Initially unassigned
        self.is_core = False  # Indicates whether the point is a core point


class DBSCRN:
    """DBSCRN clustering algorithm implementation."""
    def __init__(self, k: int):
        """
        Initializes the algorithm.
        :param k: Number of nearest neighbors to identify core points.
        """
        self.k = k
        self.points = []

    def fit(self, data: np.ndarray) -> List[List[Point]]:
        """
        Applies DBSCRN to the dataset.
        :param data: Dataset matrix (n_samples x n_features).
        :return: List of clusters.
        """
        self._initialize_points(data)
        rnn_dict = self._calculate_rnn(data)

        # Identify core points
        core_points = []
        for point, rnn in rnn_dict.items():
            if len(rnn) >= self.k:
                point.is_core = True
                core_points.append(point)

        # Expand clusters from core points
        clusters = self._expand_clusters(core_points, rnn_dict)

        # Assign remaining points to the nearest cluster
        self._assign_all_points(clusters)

        return clusters

    def _initialize_points(self, data: np.ndarray):
        """Initializes the list of points from the dataset matrix."""
        self.points = [Point(coord) for coord in data]

    def _calculate_rnn(self, data: np.ndarray) -> Dict[Point, List[Point]]:
        """
        Calculates the Reverse Nearest Neighbors (RNN) for each point.
        :param data: Dataset matrix.
        :return: Dictionary of points and their RNNs.
        """
        nbrs = NearestNeighbors(n_neighbors=self.k).fit(data)
        _, indices = nbrs.kneighbors(data)

        rnn_dict = {point: [] for point in self.points}
        for i, neighbors in enumerate(indices):
            for neighbor_idx in neighbors:
                rnn_dict[self.points[neighbor_idx]].append(self.points[i])
        return rnn_dict

    def _expand_clusters(self, core_points: List[Point], rnn_dict: Dict[Point, List[Point]]) -> List[List[Point]]:
        """
        Expands clusters starting from core points using RNN.
        :param core_points: List of core points.
        :param rnn_dict: Dictionary of points and their RNNs.
        :return: List of clusters.
        """
        clusters = []
        visited = set()

        for core in core_points:
            if core not in visited:
                cluster = []
                self._dfs_cluster(core, rnn_dict, visited, cluster)
                clusters.append(cluster)
        return clusters

    def _dfs_cluster(self, point: Point, rnn_dict: Dict[Point, List[Point]], visited: set, cluster: List[Point]):
        """
        Performs depth-first search to expand a cluster.
        :param point: Current point.
        :param rnn_dict: Dictionary of points and their RNNs.
        :param visited: Set of already visited points.
        :param cluster: The cluster being expanded.
        """
        visited.add(point)
        cluster.append(point)

        for neighbor in rnn_dict[point]:
            if neighbor.is_core and neighbor not in visited:
                self._dfs_cluster(neighbor, rnn_dict, visited, cluster)

    def _assign_all_points(self, clusters: List[List[Point]]):
        """
        Assigns all remaining points to the nearest cluster.
        :param clusters: List of already created clusters.
        """
        for point in self.points:
            if point.cluster is None:  # If the point is unassigned
                closest_cluster = None
                min_distance = float('inf')

                for cluster in clusters:
                    for core in cluster:
                        distance = np.linalg.norm(point.coordinates - core.coordinates)
                        if distance < min_distance:
                            min_distance = distance
                            closest_cluster = cluster

                # Assign the point to the nearest cluster
                if closest_cluster is not None:
                    closest_cluster.append(point)
                    point.cluster = closest_cluster

    def find_optimal_k(self, data, k_values, metrics, penalty_weight=0.5):
        """
        Finds the optimal k based on the selected metrics and penalizes excessive cluster counts.
        """
        best_k = None
        best_score = float('-inf')
        best_clusters = None
        best_labels = None

        for k in k_values:
            self.k = k
            clusters = self.fit(data)
            labels = self._get_labels(clusters)
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            # Evaluate metrics
            metrics_scores = []
            for metric in metrics:
                try:
                    score = metric(data, labels)
                except Exception as e:
                    print(f"Metric {metric.__name__} failed for k={k}: {e}")
                    score = float('-inf')
                metrics_scores.append(score)

            # Combined score with penalty for excessive clusters
            combined_score = (sum(metrics_scores) / len(metrics_scores)) - penalty_weight * num_clusters

            if combined_score > best_score:
                best_score = combined_score
                best_k = k
                best_clusters = clusters
                best_labels = labels

            print(f"k = {k}, Combined Score = {combined_score:.4f}, Metrics: {metrics_scores}, Clusters: {num_clusters}")

        return best_k, best_clusters, best_labels



    def _get_labels(self, clusters):
        """
        Assigns labels to each point based on its cluster.
        :param clusters: List of clusters (each cluster is a list of points).
        :return: Array of labels where each index corresponds to a point.
        """
        labels = -1 * np.ones(len(self.points))  # Initialize all points as noise (-1)
        for cluster_id, cluster in enumerate(clusters):
            for point in cluster:
                idx = self.points.index(point)  # Find the index of the point
                labels[idx] = cluster_id       # Assign the cluster ID as the label
        return labels

