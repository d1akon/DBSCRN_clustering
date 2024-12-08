import numpy as np
from sklearn.metrics import pairwise_distances

def gap_statistic(data, labels, n_refs=10):
    """
    Compute the Gap Statistic for evaluating the optimal number of clusters.
    :param data: Dataset to evaluate.
    :param labels: Cluster labels for the current clustering.
    :param n_refs: Number of random reference datasets to generate.
    :return: Gap statistic score.
    """
    unique_labels = np.unique(labels[labels != -1])  # Exclude noise if present
    if len(unique_labels) < 2:  # Gap Statistic requires at least two clusters
        return -np.inf

    # Compute dispersion for the original data
    orig_dispersion = compute_dispersion(data, labels)

    # Generate reference datasets and compute their dispersions
    ref_disps = []
    for _ in range(n_refs):
        random_data = np.random.uniform(low=np.min(data, axis=0), high=np.max(data, axis=0), size=data.shape)
        random_labels = np.random.choice(unique_labels, size=len(data))
        ref_dispersion = compute_dispersion(random_data, random_labels)
        ref_disps.append(ref_dispersion)

    # Calculate the Gap Statistic
    gap = np.mean(np.log(ref_disps)) - np.log(orig_dispersion)
    return gap

def compute_dispersion(data, labels):
    """
    Compute the total within-cluster dispersion for a given clustering.
    :param data: Dataset.
    :param labels: Cluster labels.
    :return: Total within-cluster dispersion.
    """
    unique_labels = np.unique(labels[labels != -1])  # Exclude noise if present
    dispersion = 0.0
    for label in unique_labels:
        cluster_points = data[labels == label]
        if len(cluster_points) > 1:  # Avoid computing for single-point clusters
            cluster_center = np.mean(cluster_points, axis=0)
            dispersion += np.sum(np.linalg.norm(cluster_points - cluster_center, axis=1))
    return dispersion
