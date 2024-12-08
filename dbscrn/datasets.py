from sklearn.datasets import make_moons, make_circles, make_blobs, make_classification
from typing import Tuple
import numpy as np

def load_datasets() -> dict:
    """
    Generates and returns a dictionary of datasets for clustering analysis.
    :return: Dictionary where keys are dataset names and values are tuples (data, true_labels).
    """
    return {
        "Moons": make_moons(n_samples=1000, noise=0.05),
        "Circles": make_circles(n_samples=1000, factor=0.5, noise=0.05),
        "Blobs": make_blobs(n_samples=1000, centers=4, cluster_std=1.0, random_state=42),
        "Classification": make_classification(
            n_samples=1000, n_features=2, n_clusters_per_class=1, n_classes=4, n_redundant=0, random_state=42
        ),
    }

def get_dataset(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieves a dataset by its name.
    :param name: Name of the dataset.
    :return: A tuple containing the dataset matrix (data) and true labels.
    :raises ValueError: If the dataset name is not found.
    """
    datasets = load_datasets()
    if name not in datasets:
        raise ValueError(f"Dataset '{name}' not found. Available datasets: {list(datasets.keys())}")
    return datasets[name]
