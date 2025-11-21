import numpy as np
from numpy.linalg import norm, matrix_rank


# ============================================================
# Solve one neuron block
# ============================================================

def solve_one_block(A, b, reg=1e-3, tol=1e-4):
    """
    Solve one neuron block Ah * w = bh using Tikhonov regularization.

    Returns:
        w: recovered weight vector
        residual: AH w - b relative residual
        full_rank: whether A has full column rank
    """
    m, n = A.shape

    ATA = A.T @ A + reg * np.eye(n)
    ATb = A.T @ b

    try:
        w = np.linalg.solve(ATA, ATb)
    except np.linalg.LinAlgError:
        return None, 1e9, False

    residual = norm(A @ w - b) / (norm(b) + 1e-8)
    full_rank = (matrix_rank(A) == n)

    return w, residual, full_rank


# ============================================================
# Main structured solver
# ============================================================

def solve_structured_system(A_blocks, b_blocks,
                            resample_callback=None,
                            max_global_iter=6,
                            tol=1e-4):
    """
    Solve structured block model Ah w = bh
    with fallback resampling or cluster switching.
    """

    hidden_dim = len(A_blocks)
    input_dim = A_blocks[0].shape[1]

    W = np.zeros((hidden_dim, input_dim))
    block_residuals = []
    rank_ok_list = []
    iterations = 0

    severe_residual_thresh = 1.0

    for h in range(hidden_dim):
        Ah = A_blocks[h]
        bh = b_blocks[h]

        for it in range(max_global_iter):

            w, res, full_rank = solve_one_block(Ah, bh)
            iterations += 1

            # Good or good-enough solution
            if w is not None and (res < tol or full_rank):
                W[h] = w
                block_residuals.append(res)
                rank_ok_list.append(full_rank)
                break

            # No callback
            if resample_callback is None:
                break

            # Severe failure: switch cluster
            if (res > severe_residual_thresh) or (not full_rank):
                Ah, bh = resample_callback(h, mode="switch_cluster")
                continue

            # Moderate: resample inside cluster
            Ah, bh = resample_callback(h, mode="resample")

        else:
            # Failed after max iterations
            return W, block_residuals, rank_ok_list, iterations, False

    return W, block_residuals, rank_ok_list, iterations, True
