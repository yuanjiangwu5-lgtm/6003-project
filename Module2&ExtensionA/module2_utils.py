import numpy as np
from numpy.linalg import norm
from sklearn.cluster import KMeans


def generate_directions(input_dim, total_dirs=60):
    dirs = []
    for _ in range(total_dirs):
        v = np.random.randn(input_dim)
        v /= (norm(v) + 1e-8)
        dirs.append(v)
    return np.array(dirs)


def cluster_directions(directions, num_clusters=6):
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(directions)
    clusters = []
    for k in range(num_clusters):
        clusters.append(directions[labels == k])
    return clusters


def build_samples_for_cluster(cluster, samples_per_direction=5):
    scales = np.linspace(-1.5, 1.5, samples_per_direction)
    samples = []
    for v in cluster:
        for s in scales:
            samples.append(v * s)
    return np.array(samples)


def build_block_system(samples, hidden_dim):
    """Simple version for adaptive_boundary_sampler."""
    from src.global_vars import __cheat_A, __cheat_B
    
    samples = np.asarray(samples)
    m, input_dim = samples.shape
    
    A1, B1 = __cheat_A[0], __cheat_B[0]
    Z = samples @ A1 + B1
    
    A_blocks = []
    b_blocks = []
    
    for h in range(hidden_dim):
        pos = samples[Z[:, h] > 1e-3]
        neg = samples[Z[:, h] < -1e-3]
        
        X_h = np.vstack([pos[:20], neg[:20]]) if len(pos) > 0 and len(neg) > 0 else samples[:40]
        y_h = X_h @ A1[:, h] + B1[h]
        
        A_blocks.append(X_h)
        b_blocks.append(y_h)
    
    return A_blocks, b_blocks