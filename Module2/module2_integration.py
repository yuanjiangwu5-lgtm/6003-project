import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from module2_structured import structured_extraction


# ================================================================
# Unified public API for Module 2 (Needed by compare_modules.py)
# ================================================================
def run_module2_extraction(architecture_str, seed, verbose=True):
    """
    Unified entry point for Module 2, required by wrapper and comparison script.
    """

    if verbose:
        print("\n===== Running Module 2 Extraction =====")
        print(f"Architecture: {architecture_str}")
        print(f"Seed: {seed}")

    architecture = [int(x) for x in architecture_str.split("-")]

    start = time.perf_counter()
    result = structured_extraction(architecture, seed, verbose=verbose)
    end = time.perf_counter()

    # Add metadata
    result["total_time"] = end - start
    result["total_iterations"] = result.get("iterations", 0)
    result["target_neurons"] = architecture[1]

    return result


# ================================================================
# Manual run for debugging
# ================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python module2_integration.py <architecture> [seed]")
        sys.exit(1)

    arch = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    out = run_module2_extraction(arch, seed, verbose=True)
    print("\n===== MODULE 2 RESULT =====")
    print(out)
