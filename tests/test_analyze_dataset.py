from utils.utils import analyze_dataset
from sklearn.datasets import make_classification, make_moons, make_blobs, make_circles
import numpy as np

np.random.seed(42)

def test_analyze_moons():
    data, _ = make_moons(n_samples=1000, noise=0.05)
    result = analyze_dataset(data)
    print(f"DEBUG - analyze_dataset for Moons: {result}")
    assert result == "density"  

def test_analyze_circles():
    data, _ = make_circles(n_samples=1000, noise=0.05)
    result = analyze_dataset(data)
    print(f"DEBUG - analyze_dataset for Moons: {result}")
    assert result == "density"  

def test_analyze_blobs():
    data, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    problem_type = analyze_dataset(data)
    assert problem_type == "mixed" or problem_type == "classic"

def test_analyze_classification():
    data, _ = make_classification(
        n_samples=1000, 
        n_features=2, 
        n_informative=2,  
        n_clusters_per_class=1, 
        n_classes=4, 
        n_redundant=0, 
        random_state=42
    )
    problem_type = analyze_dataset(data)
    assert problem_type == "mixed" or problem_type == "classic"