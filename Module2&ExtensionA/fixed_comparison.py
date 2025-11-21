"""
Fixed Module Comparison Script

Compare Module 1 (CRYPTO 2020 Baseline) vs Module 2 (EUROCRYPT 2024 Structured)
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def run_real_baseline(architecture_str, seed):
    """
    Run Module 1 baseline extraction
    """
    print("\n" + "="*70)
    print("Module 1: Random Sampling Baseline (CRYPTO 2020 style)")
    print("="*70)
    
    try:
        from module1_wrapper import run_module1_extraction
        result = run_module1_extraction(architecture_str, seed, verbose=True)
        return result
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_structured_method(architecture_str, seed):
    """
    Run Module 2 structured extraction
    """
    print("\n" + "="*70)
    print("Module 2: Structured Sampling (EUROCRYPT 2024 style)")
    print("="*70)
    
    try:
        from module2_wrapper import run_module2_extraction
        result = run_module2_extraction(architecture_str, seed, verbose=True)
        return result
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_comparison(m1_result, m2_result):
    """
    Print detailed comparison between two methods
    """
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    if m1_result is None or m2_result is None:
        print("\n✗ Cannot compare: One or both methods failed")
        return
    
    print()
    print(f"{'Metric':<30} {'Module 1':<20} {'Module 2':<20} {'Change':<15}")
    print("-" * 85)
    
    # Total Queries
    q1 = m1_result.get('total_queries', 0)
    q2 = m2_result.get('total_queries', 0)
    q_change = ((q1 - q2) / q1 * 100) if q1 > 0 else 0
    print(f"{'Total Queries':<30} {q1:<20} {q2:<20} {q_change:>+13.1f}%")
    
    # Total Time
    t1 = m1_result.get('total_time', 0)
    t2 = m2_result.get('total_time', 0)
    t_change = ((t1 - t2) / t1 * 100) if t1 > 0 else 0
    print(f"{'Total Time (s)':<30} {t1:<20.3f} {t2:<20.3f} {t_change:>+13.1f}%")
    
    # Iterations
    i1 = m1_result.get('iterations', 0)
    i2 = m2_result.get('iterations', 0)
    i_change = ((i1 - i2) / i1 * 100) if i1 > 0 else 0
    print(f"{'Iterations':<30} {i1:<20} {i2:<20} {i_change:>+13.1f}%")
    
    # Neurons Recovered
    n1 = m1_result.get('neurons_recovered', 0)
    n2 = m2_result.get('neurons_recovered', 0)
    print(f"{'Neurons Recovered':<30} {n1:<20} {n2:<20} {'=':<15}")
    
    # Efficiency (queries per neuron)
    if n1 > 0 and n2 > 0:
        eff1 = q1 / n1
        eff2 = q2 / n2
        eff_change = ((eff1 - eff2) / eff1 * 100)
        print(f"{'Efficiency (Q/neuron)':<30} {eff1:<20.1f} {eff2:<20.1f} {eff_change:>+13.1f}%")
    
    # Success
    s1 = "✓" if m1_result.get('success', False) else "✗"
    s2 = "✓" if m2_result.get('success', False) else "✗"
    print(f"{'Success':<30} {s1:<20} {s2:<20}")
    
    print("="*85)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if q_change > 0:
        print(f"✓ Module 2 uses {q_change:.1f}% fewer queries")
    elif q_change < 0:
        print(f"✗ Module 2 uses {abs(q_change):.1f}% more queries")
    else:
        print(f"= Same query count")
    
    if t_change > 0:
        print(f"✓ Module 2 is {t_change:.1f}% faster")
    elif t_change < 0:
        print(f"✗ Module 2 is {abs(t_change):.1f}% slower")
    else:
        print(f"= Same time")
    
    if eff_change > 0:
        print(f"✓ Module 2 is {eff_change:.1f}% more efficient (fewer queries per neuron)")
    elif eff_change < 0:
        print(f"✗ Module 2 is {abs(eff_change):.1f}% less efficient")
    
    print("="*80)


def save_results(m1_result, m2_result, output_dir='./results'):
    """
    Save comparison results to JSON
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if m1_result is None or m2_result is None:
        print("\n✗ Cannot save results: One or both methods failed")
        return
    
    # Calculate improvements
    q1 = m1_result.get('total_queries', 0)
    q2 = m2_result.get('total_queries', 0)
    t1 = m1_result.get('total_time', 0)
    t2 = m2_result.get('total_time', 0)
    i1 = m1_result.get('iterations', 0)
    i2 = m2_result.get('iterations', 0)
    
    improvements = {
        'queries': ((q1 - q2) / q1 * 100) if q1 > 0 else 0,
        'time': ((t1 - t2) / t1 * 100) if t1 > 0 else 0,
        'iterations': ((i1 - i2) / i1 * 100) if i1 > 0 else 0,
    }
    
    results = {
        'module1': m1_result,
        'module2': m2_result,
        'improvements': improvements
    }
    
    # Save to comparison subdirectory
    comparison_dir = Path(output_dir) / 'comparison'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = comparison_dir / 'comparison_results.json'
    with open(output_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        # Deep conversion
        results_serializable = {}
        for key, value in results.items():
            if isinstance(value, dict):
                results_serializable[key] = {k: convert(v) for k, v in value.items()}
            else:
                results_serializable[key] = convert(value)
        
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")


def main(architecture, seed):
    """
    Main comparison experiment
    """
    print("\n" + "="*80)
    print("Module 1 vs Module 2: Real Comparison Experiment")
    print("="*80)
    print(f"Architecture: {architecture}")
    print(f"Seed: {seed}\n")
    
    # Check model exists
    model_path = Path(f"models/{seed}_{architecture}.npz")
    if not model_path.exists():
        print(f"✗ Model file not found: {model_path}")
        print("  Please train the model first using: python train_models.py")
        sys.exit(1)
    
    print(f"✓ Model file found: {model_path}\n")
    
    # Step 1: Run Module 1
    print("\n" + "#"*80)
    print("# Step 1/2: Running Module 1 Baseline")
    print("#"*80)
    m1 = run_real_baseline(architecture, seed)
    
    # Reset for fair comparison
    from src import global_vars
    global_vars.query_count = 0
    
    # Step 2: Run Module 2
    print("\n" + "#"*80)
    print("# Step 2/2: Running Module 2 Structured")
    print("#"*80)
    m2 = run_structured_method(architecture, seed)
    
    # Compare results
    print_comparison(m1, m2)
    
    # Save results
    save_results(m1, m2)
    
    print("\n" + "="*80)
    print("✓ Comparison Experiment Complete")
    print("="*80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fixed_comparison.py <architecture> [seed]")
        print("Example: python fixed_comparison.py 10-10-1 42")
        sys.exit(1)
    
    architecture = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    
    try:
        main(architecture, seed)
    except KeyboardInterrupt:
        print("\n\n⊘ Comparison interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)