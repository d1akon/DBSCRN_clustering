import numpy as np
from scipy.spatial.distance import cdist

def dunn(data: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculates the Dunn Index to evaluate the quality of clusters.
    
    :param data: Data matrix (n_samples x n_features).
    :param labels: Cluster labels (n_samples,).
    :return: Dunn Index (float).
    """
    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters)
    
    if n_clusters < 2:
        return -1  # Cannot calculate if there are fewer than 2 clusters
    
    #----- Calculate the diameters of the clusters
    cluster_diameters = []
    for cluster in unique_clusters:
        points_in_cluster = data[labels == cluster]
        if len(points_in_cluster) > 1:
            pairwise_distances = cdist(points_in_cluster, points_in_cluster)
            cluster_diameters.append(pairwise_distances.max())
        else:
            cluster_diameters.append(0)  # If the cluster has only one point, diameter is 0

    max_diameter = max(cluster_diameters)

    #----- Calculate distances between clusters
    inter_cluster_distances = []
    for i, cluster_i in enumerate(unique_clusters):
        points_in_cluster_i = data[labels == cluster_i]
        for j, cluster_j in enumerate(unique_clusters):
            if i < j:
                points_in_cluster_j = data[labels == cluster_j]
                pairwise_distances = cdist(points_in_cluster_i, points_in_cluster_j)
                inter_cluster_distances.append(pairwise_distances.min())

    min_inter_cluster_distance = min(inter_cluster_distances)

    #----- Calculate the Dunn Index
    if max_diameter > 0:
        dunn_index = min_inter_cluster_distance / max_diameter
    else:
        dunn_index = -1  #--- It's not valid if the maximum diameter is 0
    
    return dunn_index
