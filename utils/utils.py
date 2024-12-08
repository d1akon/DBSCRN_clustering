from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import cdist
from dbscrn.dunn_index import dunn
import numpy as np

def classify_problem(density_score, convexity_score, dunn_index, davies_bouldin, density_proportion):
    """
    Clasifica el tipo de problema basado en métricas clave con umbrales refinados.
    """
    if dunn_index < 0.01 and density_score > 1.0 and davies_bouldin > 0.7:
        return "density"
    elif convexity_score > 500 and density_proportion < 0.8:
        return "classic"
    elif 0.01 <= dunn_index < 0.2 and density_proportion < 1:
        return "mixed"
    else:
        return "classic"





def analyze_dataset(data, k=10):
    """
    Analyzes the dataset to determine the type of clustering problem.
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
    from scipy.spatial.distance import cdist

    nbrs = NearestNeighbors(n_neighbors=k).fit(data)
    distances, _ = nbrs.kneighbors(data)
    mean_density = distances[:, 1].mean()
    std_density = distances[:, 1].std()
    dispersion = np.mean(cdist(data, data).flatten())

    # Generate valid labels with KMeans
    kmeans = KMeans(n_clusters=2, random_state=42).fit(data)
    valid_labels = kmeans.labels_

    # Metrics
    convexity = calinski_harabasz_score(data, valid_labels)
    davies_bouldin = davies_bouldin_score(data, valid_labels)
    density_ratio = std_density / mean_density
    dunn_index = dunn(data, valid_labels)

    # Log metrics
    print(f"Density (mean, std): {mean_density:.4f}, {std_density:.4f}")
    print(f"Global Dispersion: {dispersion:.4f}")
    print(f"Estimated Convexity: {convexity:.4f}")
    print(f"Davies-Bouldin: {davies_bouldin:.4f}")
    print(f"Dunn Index: {dunn_index:.4f}")
    print(f"Density Proportion (std/mean): {density_ratio:.4f}")

    # Scores
    density_score = (1 / density_ratio) * davies_bouldin
    convexity_score = convexity / dispersion

    print(f"Density Score: {density_score:.4f}, Convexity Score: {convexity_score:.4f}")

    # Classify problem
    problem_type = classify_problem(density_score, convexity_score, dunn_index, davies_bouldin, density_ratio)
    return problem_type
