"""
Unified comparison experiment: Module 1 vs Module 2

Uses wrapper interfaces for a fair comparison.
"""
"""
Unified comparison experiment: Module 1 vs Module 2
Uses wrapper interfaces for a fair comparison.
"""

import sys
import json
import numpy as np
import time
from pathlib import Path     

ROOT = str(Path(__file__).resolve().parent.parent)
SRC = str(Path(__file__).resolve().parent)
sys.path.insert(0, ROOT)
sys.path.insert(0, SRC)

def load_wrappers():
    try:
        from module2.wrapper import run_module2_extraction
        from module1.wrapper import run_module1_extraction
        return run_module1_extraction, run_module2_extraction
    except ImportError:
        print("Using fallback root-level wrappers...")
        from module1_wrapper import run_module1_extraction
        from module2_wrapper import run_module2_extraction
        return run_module1_extraction, run_module2_extraction

def print_comparison(m1, m2):
    """Formatted comparison between Module 1 and Module 2."""
    print("\n" + "="*80)
    print("Module 1 vs Module 2 Comparison")
    print("="*80)
    print()
    
    print(f"{'Metric':<30} {'Module 1 (CRYPTO 2020)':<20} "
          f"{'Module 2 (EUROCRYPT 2024)':<20} {'Improvement':<15}")
    print("-" * 80)
    
    # Neuron recovery
    print(f"{'Neurons Recovered':<30} "
          f"{m1['neurons_recovered']}/{m1['target_neurons']:<18} "
          f"{m2['neurons_recovered']}/{m2['target_neurons']:<18} "
          f"{'-':<15}")

    # Total queries
    q1, q2 = m1['total_queries'], m2['total_queries']
    if q1 > 0:
        imp_q = (q1 - q2) / q1 * 100
        print(f"{'Total Queries':<30} {q1:<20} {q2:<20} {imp_q:>13.1f}%")
    else:
        print(f"{'Total Queries':<30} {q1:<20} {q2:<20} {'N/A':<15}")

    # Total time
    t1, t2 = m1['total_time'], m2['total_time']
    if t1 > 0:
        imp_t = (t1 - t2) / t1 * 100
        print(f"{'Total Time (s)':<30} {t1:<20.2f} {t2:<20.2f} {imp_t:>13.1f}%")
    else:
        print(f"{'Total Time (s)':<30} {t1:<20.2f} {t2:<20.2f} {'N/A':<15}")

    # Iterations
    i1 = m1.get('iterations', 0)
    i2 = m2.get('total_iterations', 0)
    if i1 > 0:
        imp_i = (i1 - i2) / i1 * 100
        print(f"{'Iterations':<30} {i1:<20} {i2:<20} {imp_i:>13.1f}%")
    else:
        print(f"{'Iterations':<30} {i1:<20} {i2:<20} {'N/A':<15}")

    # Queries per neuron (efficiency)
    nr1 = m1["neurons_recovered"]
    nr2 = m2["neurons_recovered"]

    eff1 = q1 / nr1 if nr1 > 0 else None
    eff2 = q2 / nr2 if nr2 > 0 else None

    if eff1 is None or eff2 is None:
        print(f"{'Queries per Neuron':<30} {'N/A':<20} {'N/A':<20} {'N/A':<15}")
    elif eff1 == 0:
        # Baseline uses 0 queries → cannot define relative improvement
        print(f"{'Queries per Neuron':<30} {eff1:<20.1f} {eff2:<20.1f} "
              f"{'N/A (baseline=0)':<15}")
    else:
        imp_eff = (eff1 - eff2) / eff1 * 100
        print(f"{'Queries per Neuron':<30} {eff1:<20.1f} {eff2:<20.1f} "
              f"{imp_eff:>13.1f}%")

    # Success flags
    s1 = "✓" if m1.get('success', False) else "✗"
    s2 = "✓" if m2.get('success', False) else "✗"
    print(f"{'Success':<30} {s1:<20} {s2:<20} {'-':<15}")

    print("=" * 80)


def save_results(m1, m2, output_dir='./results/comparison'):
    """Save comparison results to JSON"""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Fix: convert numpy arrays into JSON serializable format
    def to_serializable(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return obj

    improvements = {}
    if m1['total_queries'] > 0 and m2['total_queries'] > 0:
        improvements['queries'] = (m1['total_queries'] - m2['total_queries']) / m1['total_queries'] * 100
    else:
        improvements['queries'] = None

    if m1['total_time'] > 0:
        improvements['time'] = (m1['total_time'] - m2['total_time']) / m1['total_time'] * 100
    else:
        improvements['time'] = None

    i1, i2 = m1.get('iterations', 0), m2.get('total_iterations', 0)
    if i1 > 0 and i2 > 0:
        improvements['iterations'] = (i1 - i2) / i1 * 100
    else:
        improvements['iterations'] = None

    results = {
        'module1': m1,
        'module2': m2,
        'improvements': improvements,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    output_path = Path(output_dir) / 'comparison_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=to_serializable)

    print(f"\nResults saved to: {output_path}")

    return output_path



def main(architecture, seed):
    print("\n" + "="*80)
    print("Unified Comparison Framework")
    print("="*80)
    print(f"Architecture: {architecture}")
    print(f"Seed: {seed}")
    print()

    # Load wrappers
    print("Loading wrappers...")
    run_module1, run_module2 = load_wrappers()
    print("✓ Wrappers loaded successfully")

    # Module 1
    print("\n" + "#"*80)
    print("# Step 1/2: Running Module 1 (CRYPTO 2020)")
    print("#"*80)
    m1 = run_module1(architecture, seed, verbose=True)

    # Module 2
    print("\n" + "#"*80)
    print("# Step 2/2: Running Module 2 (EUROCRYPT 2024)")
    print("#"*80)
    m2 = run_module2(architecture, seed, verbose=True)

    # Print comparison
    print_comparison(m1, m2)

    # Save results
    save_results(m1, m2)

    print("\n" + "="*80)
    print("Comparison Complete")
    print("="*80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_modules.py <architecture> [seed]")
        sys.exit(1)

    arch = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    main(arch, seed)