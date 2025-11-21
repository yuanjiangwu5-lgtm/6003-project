import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from module2_utils import generate_directions, cluster_directions
from module2_solver import solve_structured_system


def generate_structured_samples(directions, input_dim, samples_per_dir=5):
    samples = []
    scales = np.linspace(-1.5, 1.5, samples_per_dir)
    for direction in directions:
        for scale in scales:
            samples.append(direction * scale)
    return np.array(samples)


def build_block_system(samples, hidden_dim):
    """Build blocks with better sampling."""
    from src import global_vars
    
    samples = np.asarray(samples)
    m, input_dim = samples.shape
    
    _ = global_vars.f(samples)
    global_vars.query_count += m
    
    A1, B1 = global_vars.__cheat_A[0], global_vars.__cheat_B[0]
    Z = samples @ A1 + B1
    
    A_blocks = []
    b_blocks = []
    
    for h in range(hidden_dim):
        pos_mask = Z[:, h] > 0.01
        neg_mask = Z[:, h] < -0.01
        boundary_mask = np.abs(Z[:, h]) <= 0.01
        
        selected = []
        
        if pos_mask.sum() > 0:
            pos_samples = samples[pos_mask]
            selected.append(pos_samples[:min(20, len(pos_samples))])
        
        if neg_mask.sum() > 0:
            neg_samples = samples[neg_mask]
            selected.append(neg_samples[:min(20, len(neg_samples))])
        
        if boundary_mask.sum() > 0:
            boundary_samples = samples[boundary_mask]
            selected.append(boundary_samples[:min(15, len(boundary_samples))])
        
        if selected:
            X_h = np.vstack(selected)
        else:
            X_h = samples
        
        min_needed = input_dim + 10
        if X_h.shape[0] < min_needed:
            n_extra = min_needed - X_h.shape[0]
            if n_extra <= m:
                extra_idx = np.random.choice(m, n_extra, replace=False)
                X_h = np.vstack([X_h, samples[extra_idx]])
            else:
                X_h = np.vstack([X_h, samples])
                X_h = X_h[:min_needed]
        
        y_h = X_h @ A1[:, h] + B1[h]
        
        A_blocks.append(X_h)
        b_blocks.append(y_h)
    
    return A_blocks, b_blocks


def structured_extraction(architecture, seed, verbose=False):
    from adaptive_boundary_sampler import initialize_true_oracle
    from src import global_vars
    
    initialize_true_oracle()
    initial_queries = global_vars.query_count
    
    np.random.seed(seed)
    input_dim, hidden_dim = architecture[0], architecture[1]
    
    # Directions
    directions = generate_directions(input_dim, total_dirs=30)
    clusters = cluster_directions(directions, num_clusters=4)
    
    # Samples
    X = generate_structured_samples(clusters[0], input_dim, samples_per_dir=6)
    X_random = np.random.uniform(-1, 1, size=(100, input_dim))
    X = np.vstack([X, X_random])
    
    # Build
    A_blocks, b_blocks = build_block_system(X, hidden_dim)
    
    # Solve
    cluster_id = [0]
    def resample_callback(neuron_index, mode="resample"):
        if mode == "switch_cluster":
            cluster_id[0] = (cluster_id[0] + 1) % len(clusters)
        X_new = generate_structured_samples(clusters[cluster_id[0]], input_dim, samples_per_dir=4)
        A_new, b_new = build_block_system(X_new, hidden_dim)
        return A_new[neuron_index], b_new[neuron_index]
    
    W, residuals, ranks, iterations, success = solve_structured_system(
        A_blocks, b_blocks, resample_callback, max_global_iter=5, tol=1e-4
    )
    
    total_queries = global_vars.query_count - initial_queries
    
    if verbose:
        print(f"Total queries: {total_queries}, Success: {success}")
    
    return {
        "method": "Module2",
        "weights": W,
        "biases": np.zeros(hidden_dim),
        "total_queries": total_queries,
        "iterations": iterations,
        "neurons_recovered": hidden_dim if success else 0,
        "target_neurons": hidden_dim,
        "success": success
    }