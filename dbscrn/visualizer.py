import matplotlib.pyplot as plt
import numpy as np
from typing import List
from .algorithm import Point

class Visualizer:
    """Class for visualizing clustering results."""
    @staticmethod
    def plot_clusters(data: np.ndarray, clusters: List[List[Point]]):
        """
        Generates a scatter plot of the detected clusters.
        :param data: Original dataset matrix.
        :param clusters: List of detected clusters.
        """
        colors = plt.cm.rainbow(np.linspace(0, 1, len(clusters)))
        for cluster, color in zip(clusters, colors):
            cluster_data = np.array([point.coordinates for point in cluster])
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], color=color, label=f"Cluster {clusters.index(cluster)}")

        plt.title("Detected Clusters")
        plt.legend()
        plt.show()
