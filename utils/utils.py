from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import cdist
from dbscrn.dunn_index import dunn
import numpy as np

def classify_problem(density_score, convexity_score, dunn_index, davies_bouldin, density_proportion):
    """
    Classifies the type of clustering problem based on refined thresholds.
    """
    print("\n-- Problem Classification Debugging --")
    print(f"Density Score: {density_score:.4f}, Convexity Score: {convexity_score:.4f}")
    print(f"Dunn Index: {dunn_index:.4f}, Davies-Bouldin: {davies_bouldin:.4f}")
    print(f"Density Proportion: {density_proportion:.4f}")

    if dunn_index < 0.02 and density_score > 0.7 and density_proportion < 0.8:
        decision = "density"
    elif convexity_score > 400 and dunn_index > 0.03 and density_proportion > 0.9:
        decision = "classic"
    elif 0.02 <= dunn_index < 0.05 and 0.8 <= density_proportion <= 1.2:
        decision = "mixed"
    else:
        #---- Fallback logic based on density score
        decision = "density" if density_score > 0.6 else "classic"

    print(f"Decision: {decision}")
    return decision

def analyze_dataset(data, k=10, max_clusters=10, use_dynamic_clusters=True):
    """
    Analyzes the dataset to determine the type of clustering problem.
    
    :param data: Input dataset (n_samples, n_features).
    :param k: Number of neighbors for density-based metrics.
    :param max_clusters: Maximum number of clusters to consider if use_dynamic_clusters is True.
    :param use_dynamic_clusters: Whether to dynamically determine the optimal number of clusters.
    :return: Estimated problem type (e.g., "classic", "density", "mixed").
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
    from scipy.spatial.distance import cdist
    import numpy as np

    #----- Step 1: Density-based metrics
    nbrs = NearestNeighbors(n_neighbors=k).fit(data)
    distances, _ = nbrs.kneighbors(data)
    mean_density = distances[:, 1].mean()
    std_density = distances[:, 1].std()
    dispersion = np.mean(cdist(data, data).flatten())

    #----- Step 2: Determine clusters using KMeans
    if use_dynamic_clusters:
        scores = []
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data)
            labels = kmeans.labels_
            score = calinski_harabasz_score(data, labels)
            scores.append((n_clusters, score))
        best_n_clusters = max(scores, key=lambda x: x[1])[0]
    else:
        best_n_clusters = 2

    #--- Generate labels with the best number of clusters
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42).fit(data)
    valid_labels = kmeans.labels_

    #----- Step 3: Calculate metrics
    convexity = calinski_harabasz_score(data, valid_labels)
    davies_bouldin = davies_bouldin_score(data, valid_labels)
    density_ratio = std_density / mean_density
    dunn_index = dunn(data, valid_labels)

    #----- Step 4: Log metrics
    print(f"Density (mean, std): {mean_density:.4f}, {std_density:.4f}")
    print(f"Global Dispersion: {dispersion:.4f}")
    print(f"Estimated Convexity: {convexity:.4f}")
    print(f"Davies-Bouldin: {davies_bouldin:.4f}")
    print(f"Dunn Index: {dunn_index:.4f}")
    print(f"Density Proportion (std/mean): {density_ratio:.4f}")
    print(f"Optimal number of clusters: {best_n_clusters}")

    # Step 5: Scores
    density_score = (1 / density_ratio) * davies_bouldin
    convexity_score = convexity / dispersion

    print(f"Density Score: {density_score:.4f}, Convexity Score: {convexity_score:.4f}")

    # Step 6: Classify problem
    problem_type = classify_problem(
        density_score,
        convexity_score,
        dunn_index,
        davies_bouldin,
        density_ratio
    )
    return problem_type
