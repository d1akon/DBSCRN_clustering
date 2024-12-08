import numpy as np
from scipy.spatial.distance import cdist

def dunn(data: np.ndarray, labels: np.ndarray) -> float:
    """
    Calcula el Dunn Index para evaluar la calidad de los clusters.
    
    :param data: Matriz de datos (n_samples x n_features).
    :param labels: Etiquetas de los clusters (n_samples,).
    :return: Dunn Index (float).
    """
    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters)
    
    if n_clusters < 2:
        return -1  # No se puede calcular si hay menos de 2 clusters
    
    # Calcular diámetros de los clusters
    cluster_diameters = []
    for cluster in unique_clusters:
        points_in_cluster = data[labels == cluster]
        if len(points_in_cluster) > 1:
            pairwise_distances = cdist(points_in_cluster, points_in_cluster)
            cluster_diameters.append(pairwise_distances.max())
        else:
            cluster_diameters.append(0)  # Si el cluster tiene un solo punto, diámetro es 0

    max_diameter = max(cluster_diameters)

    # Calcular distancias entre clusters
    inter_cluster_distances = []
    for i, cluster_i in enumerate(unique_clusters):
        points_in_cluster_i = data[labels == cluster_i]
        for j, cluster_j in enumerate(unique_clusters):
            if i < j:
                points_in_cluster_j = data[labels == cluster_j]
                pairwise_distances = cdist(points_in_cluster_i, points_in_cluster_j)
                inter_cluster_distances.append(pairwise_distances.min())

    min_inter_cluster_distance = min(inter_cluster_distances)

    # Calcular el índice Dunn
    if max_diameter > 0:
        dunn_index = min_inter_cluster_distance / max_diameter
    else:
        dunn_index = -1  # No es válido si el diámetro máximo es 0
    
    return dunn_index