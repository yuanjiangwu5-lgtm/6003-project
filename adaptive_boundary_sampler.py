"""
Adaptive Boundary Sampler for Module 2
========================================
Implements error-driven adaptive sampling to reduce query cost
and improve extraction stability. (Extension A)
"""

import os
import numpy as np
from numpy.linalg import norm
import sys
from pathlib import Path
from module2_utils import build_block_system

sys.path.insert(0, str(Path(__file__).parent))

from src import global_vars
from src.find_witnesses import do_better_sweep

print("THIS FILE IS RUNNING:", __file__)

print("[DEBUG] adaptive_boundary_sampler loaded.")


# ============================================================
# Initialize ground-truth oracle
# ============================================================

def initialize_true_oracle():
    """
    Assign true oracle (from __cheat_A/B)
    """
    from src.global_vars import __cheat_A, __cheat_B

    def oracle(x):
        h = x
        for i, (a, b) in enumerate(zip(__cheat_A, __cheat_B)):
            h = h @ a + b
            if i < len(__cheat_A) - 1:
                h = np.maximum(h, 0)
        return h

    global_vars.f = oracle
    print("[OK] True oracle initialized.")


# ============================================================
# Query wrapper
# ============================================================

def query_oracle(X):
    if X.ndim == 1:
        X = X.reshape(1, -1)
    global_vars.query_count += X.shape[0]
    return global_vars.f(X)


# ============================================================
# Recovered model prediction
# ============================================================

def recovered_model_predict(X, W, b):
    return np.maximum(0, X @ W.T + b)


# ============================================================
# Boundary sweep sampler
# ============================================================

def sample_boundary_points(direction, offset=None, num_points=5,
                           low=-1e3, high=1e3, max_queries_per_dir=100):

    if offset is None:
        offset = np.random.randn(global_vars.DIM)

    start_q = global_vars.query_count

    pts = do_better_sweep(
        offset=offset,
        direction=direction,
        low=low,
        high=high,
        return_upto_one=False
    )

    used = global_vars.query_count - start_q

    if len(pts) > num_points:
        idx = np.linspace(0, len(pts) - 1, num_points, dtype=int)
        pts = [pts[i] for i in idx]

    return pts, used


# ============================================================
# Error hotspot detection
# ============================================================

def detect_error_hotspots(W, b, dirs, num_test_points=10, threshold=0.1):

    errors = []

    for v in dirs:
        scales = np.linspace(-2, 2, num_test_points)
        P = np.array([v * s for s in scales])

        y_true = query_oracle(P)
        y_pred = recovered_model_predict(P, W, b)

        err = np.mean(np.abs(y_true - y_pred))
        errors.append(err)

    errors = np.array(errors)
    idx = np.where(errors > threshold)[0]
    return errors, idx


# ============================================================
# Adaptive resampling
# ============================================================

def adaptive_resample(priority_dirs, X_current,
                      query_budget_remaining, samples_per_dir=5):

    new_pts = []
    used = 0

    for v in priority_dirs:
        if used >= query_budget_remaining:
            break

        pts, q = sample_boundary_points(
            direction=v,
            num_points=samples_per_dir,
            max_queries_per_dir=query_budget_remaining - used
        )
        new_pts.extend(pts)
        used += q

    return np.array(new_pts), used

# ============================================================
# Main Adaptive Structured Extraction (Extension A)
# ============================================================

def adaptive_structured_extraction(architecture, seed,
                                   query_budget=5000,
                                   adaptive_rounds=3,
                                   verbose=True):

    initialize_true_oracle()
    np.random.seed(seed)
    global_vars.query_count = 0

    input_dim = architecture[0]
    hidden_dim = architecture[1]

    from module2_utils import generate_directions
    initial_dirs = generate_directions(input_dim, total_dirs=30)

    # ------------------------------
    # Phase 1: Initial sampling
    # ------------------------------
    samples = []
    for v in initial_dirs[:10]:
        pts, _ = sample_boundary_points(
            direction=v,
            num_points=3,
            max_queries_per_dir=query_budget // 20
        )
        samples.extend(pts)
        if global_vars.query_count >= query_budget * 0.3:
            break

    X_train = np.array(samples)

    # ------------------------------
    # Phase 2: Initial recovery
    # ------------------------------
    from module2_utils import build_block_system
    A_blocks, B_blocks = build_block_system(X_train, hidden_dim)


    from module2_solver import solve_structured_system
    W, resids, ranks, iters, success = solve_structured_system(
        A_blocks, B_blocks,
        resample_callback=None,
        max_global_iter=3,
        tol=1e-4
    )

    b = np.zeros(hidden_dim)

    # ------------------------------
    # Phase 3: Adaptive refinement
    # ------------------------------
    metrics = []

    hotspot_thresh = 0.05
    severe_thresh = 0.5

    for rd in range(adaptive_rounds):

        if global_vars.query_count >= query_budget:
            break

        test_dirs = generate_directions(input_dim, total_dirs=20)

        err_scores, hot_idx = detect_error_hotspots(
            W, b,
            test_dirs,
            num_test_points=5,
            threshold=hotspot_thresh
        )

        if len(hot_idx) == 0:
            if verbose:
                print(f"[Round {rd+1}] No hotspots. Stopping.")
            break

        max_err = float(np.max(err_scores))
        mean_err = float(np.mean(err_scores))

        if max_err > severe_thresh:
            samples_per_dir = 6
            if verbose:
                print(f"[Round {rd+1}] Severe error detected → Dense sampling.")
        else:
            samples_per_dir = 3

        priority_dirs = test_dirs[hot_idx]
        budget_left = query_budget - global_vars.query_count

        new_pts, q_used = adaptive_resample(
            priority_dirs,
            X_train,
            query_budget_remaining=budget_left,
            samples_per_dir=samples_per_dir
        )

        if new_pts.size == 0:
            break

        X_train = np.vstack([X_train, new_pts])
        A_blocks, B_blocks = build_block_system(X_train, hidden_dim)

        W, resids, ranks, iters, success = solve_structured_system(
            A_blocks, B_blocks,
            resample_callback=None,
            max_global_iter=2,
            tol=1e-4
        )

        metrics.append({
            "round": rd + 1,
            "queries": int(global_vars.query_count),
            "samples": int(X_train.shape[0]),
            "mean_error": mean_err,
            "max_error": max_err,
            "hotspot_count": int(len(hot_idx)),
            "mean_residual": float(np.mean(resids)),
        })

    return {
        "method": "Module2_Adaptive",
        "weights": W,
        "biases": b,
        "total_queries": int(global_vars.query_count),
        "total_samples": int(X_train.shape[0]),
        "adaptive_rounds_completed": len(metrics),
        "metrics_log": metrics,
        "final_residual": float(np.mean(resids)),
        "success": success
    }

# ============================================================
# Program Entry
# ============================================================

if __name__ == "__main__":

    print("[DEBUG] ENTERED MAIN BLOCK.")

    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python adaptive_boundary_sampler.py <arch> [seed] [mode]")
        sys.exit(1)

    arch = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    mode = sys.argv[3] if len(sys.argv) > 3 else "adaptive"

    architecture = [int(x) for x in arch.split("-")]

    print("[DEBUG] Parsed args:",
          "arch =", architecture,
          "seed =", seed,
          "mode =", mode)

    if mode == "compare":
        from compare_modules import compare_baseline_vs_adaptive
        print("[DEBUG] Running compare mode...")
        out = compare_baseline_vs_adaptive(arch, seed)
        print(out)

    else:
        print("[DEBUG] Running ADAPTIVE extraction...")
        res = adaptive_structured_extraction(
            architecture,
            seed,
            query_budget=5000,
            adaptive_rounds=3,
            verbose=True
        )
        print("\n===== FINAL RESULT =====")
        print(res)

