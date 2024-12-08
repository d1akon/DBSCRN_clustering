# DBSCRN_clustering
Density-Based Shared Clustering for Reverse Nearest Neighbors Python Implementation

Based on this paper: https://arxiv.org/abs/1811.07615

This repository implements DBSCRN along with tools for dataset analysis, cluster visualization, and advanced metric evaluations, making it a versatile solution for clustering tasks.

### Key Features

* Hybrid Clustering Capability: Unlike DBSCAN, DBSCRN adapts to both density-based and convex-shaped datasets, allowing it to handle datasets like Moons, Circles, Blobs, and Classification seamlessly.

* Reverse Nearest Neighbors (RNN): The algorithm uses RNN for enhanced core point identification and cluster formation, which improves cluster detection in challenging datasets.

* Automatic k-Optimization: Supports finding the optimal k (number of neighbors) for clustering by evaluating multiple metrics, including:
  * Silhouette Score
  * Davies-Bouldin Index
  * Dunn Index
  * Gap Statistic
  * Calinski-Harabasz Score
  * Custom Dataset Analysis:

* Includes an analyze_dataset function to classify datasets as density-based, classic, or mixed and select appropriate metrics accordingly.


Some results obtained by applying it to both density-based problems and more classical clustering problems.

![image](https://github.com/user-attachments/assets/0dbef1a8-4dcf-4b3b-8e0b-76d749c82638)
![image](https://github.com/user-attachments/assets/6829dd06-74b0-446d-9d72-f20975494c3d)
![image](https://github.com/user-attachments/assets/9613559c-e4be-4f8c-b2c9-fe876706ff88)
![image](https://github.com/user-attachments/assets/8722e58e-aad3-4ebd-9e07-d855d89ad09b)




