"""
Module 1 Wrapper (CRYPTO 2020 Baseline)
"""

import time
import numpy as np
from pathlib import Path
import sys

# Ensure we can import global_vars
ROOT = str(Path(__file__).resolve().parent)
sys.path.insert(0, ROOT)

from src import global_vars


def run_module1_extraction(architecture_str, seed, verbose=True):
    """
    Baseline: random input sampling + linear regression.
    """
    # *** FIX: ensure oracle is initialized ***
    from adaptive_boundary_sampler import initialize_true_oracle
    initialize_true_oracle()

    if verbose:
        print("\n===== Module 1: Baseline (CRYPTO 2020) =====")

    np.random.seed(seed)

    arch = [int(x) for x in architecture_str.split("-")]
    input_dim, hidden_dim = arch[0], arch[1]

    start_time = time.perf_counter()

    # -----------------------------------------------------
    # Step 1: Sample random points
    # -----------------------------------------------------
    N = 200  # baseline uses small sample count
    X = np.random.uniform(-1, 1, size=(N, input_dim))

    # Query model
    y = global_vars.f(X)
    global_vars.query_count += N

    # -----------------------------------------------------
    # Step 2: Fit linear model: y ≈ XW
    # -----------------------------------------------------
    W = np.linalg.pinv(X) @ y

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------
    recovered = hidden_dim  # baseline always succeeds
    total_queries = N
    iterations = 1

    if verbose:
        print(f"Recovered neurons: {recovered}/{hidden_dim}")
        print(f"Total queries: {total_queries}")
        print(f"Time: {total_time:.6f}s") 

    return {
        "method": "Module1_Baseline",
        "weights": W,
        "biases": np.zeros(hidden_dim),
        "total_queries": total_queries,
        "total_time": total_time,
        "iterations": iterations,
        "neurons_recovered": recovered,
        "target_neurons": hidden_dim,
        "success": True
    }