from utils.utils import analyze_dataset
from sklearn.datasets import make_classification, make_moons, make_blobs, make_circles
import numpy as np

np.random.seed(42)

def test_analyze_moons():
    from sklearn.datasets import make_moons
    data, _ = make_moons(n_samples=500, noise=0.05)
    result = analyze_dataset(data)
    print(f"Test Result for Moons: {result}")
    assert result == "density"

def test_analyze_circles():
    from sklearn.datasets import make_circles
    data, _ = make_circles(n_samples=500, factor=0.5, noise=0.05)
    result = analyze_dataset(data)
    print(f"Test Result for Circles: {result}")
    assert result == "density"

def test_analyze_blobs():
    from sklearn.datasets import make_blobs
    data, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)
    result = analyze_dataset(data)
    print(f"Test Result for Blobs: {result}")
    assert result == "classic"

def test_analyze_classification():
    from sklearn.datasets import make_classification
    data, _ = make_classification(n_samples=500, n_features=2, n_clusters_per_class=1, n_classes=4, n_redundant=0)
    result = analyze_dataset(data)
    print(f"Test Result for Classification: {result}")
    assert result == "mixed" or result == "density"
