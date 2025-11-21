"""
Robustness Testing: Evaluate Module 1 and Module 2 under noise and quantization
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))


def add_noise_to_oracle(noise_level=0.01):
    """
    Inject Gaussian noise into the global oracle.

    Returns:
        original_f: the original oracle function (so it can be restored later)
    """
    from src import global_vars

    original_f = global_vars.f

    def noisy_f(x):
        output = original_f(x)
        noise = np.random.normal(0, noise_level, output.shape)
        return output + noise

    global_vars.f = noisy_f
    return original_f


def restore_oracle(original_f):
    """Restore original oracle function"""
    from src import global_vars
    global_vars.f = original_f


def test_module_with_noise(module_extractor, architecture_str, seed, noise_level):
    """
    Test a single module under a given noise level.

    Args:
        module_extractor: extraction function (module1 or module2 wrapper)
        architecture_str: architecture string
        seed: random seed
        noise_level: injected noise magnitude

    Returns:
        dict containing performance metrics
    """
    from adaptive_boundary_sampler import initialize_true_oracle
    from src import global_vars

    initialize_true_oracle()

    # Add noise if needed
    original_f = None
    if noise_level > 0:
        original_f = add_noise_to_oracle(noise_level)

    global_vars.query_count = 0

    try:
        start_time = time.perf_counter()

        result = module_extractor(architecture_str, seed, verbose=False)

        end_time = time.perf_counter()
        actual_time = end_time - start_time

        if original_f is not None:
            restore_oracle(original_f)

        return {
            'noise_level': noise_level,
            'success': result.get('success', False),
            'queries': result.get('total_queries', 0),
            'time': actual_time,
            'neurons_recovered': result.get('neurons_recovered', 0)
        }

    except Exception as e:
        print(f"  ✗ Error at noise level {noise_level}: {e}")

        if original_f is not None:
            restore_oracle(original_f)

        return {
            'noise_level': noise_level,
            'success': False,
            'queries': 0,
            'time': 0,
            'neurons_recovered': 0
        }


def test_module1_robustness(architecture_str, seed, noise_levels=[0, 0.01, 0.05, 0.1]):
    """Run robustness test for Module 1 (CRYPTO 2020)"""
    print("\n" + "="*70)
    print("Testing Module 1 Robustness")
    print("="*70)

    from module1_wrapper import run_module1_extraction

    results = []
    for noise in noise_levels:
        print(f"\n  Noise level: {noise}")
        result = test_module_with_noise(run_module1_extraction, architecture_str, seed, noise)
        results.append(result)

        if result['success']:
            print(f"  ✓ Success - Queries: {result['queries']}, Time: {result['time']:.3f}s")
        else:
            print(f"  ✗ Failed")

    return results


def test_module2_robustness(architecture_str, seed, noise_levels=[0, 0.01, 0.05, 0.1]):
    """Run robustness test for Module 2 (EUROCRYPT 2024)"""
    print("\n" + "="*70)
    print("Testing Module 2 Robustness")
    print("="*70)

    from module2_wrapper import run_module2_extraction

    results = []
    for noise in noise_levels:
        print(f"\n  Noise level: {noise}")
        result = test_module_with_noise(run_module2_extraction, architecture_str, seed, noise)
        results.append(result)

        if result['success']:
            print(f"  ✓ Success - Queries: {result['queries']}, Time: {result['time']:.3f}s")
        else:
            print(f"  ✗ Failed")

    return results


def print_robustness_comparison(module1_results, module2_results):
    """Print robustness comparison between Module 1 and Module 2"""
    print("\n" + "="*80)
    print("Robustness Test Results")
    print("="*80)
    print()

    print(f"{'Noise Level':<15} {'Module 1 Success':<20} {'Module 2 Success':<20} {'Difference':<15}")
    print("-" * 80)

    for m1, m2 in zip(module1_results, module2_results):
        noise = m1['noise_level']
        s1 = 100 if m1['success'] else 0
        s2 = 100 if m2['success'] else 0
        diff = s2 - s1
        print(f"{noise:<15.2f} {s1:<20.1f}% {s2:<20.1f}% {diff:>+13.1f}%")

    print()
    print(f"{'Noise Level':<15} {'Module 1 Queries':<20} {'Module 2 Queries':<20} {'Improvement':<15}")
    print("-" * 80)

    for m1, m2 in zip(module1_results, module2_results):
        noise = m1['noise_level']
        q1 = m1['queries']
        q2 = m2['queries']
        if q1 > 0:
            improvement = (q1 - q2) / q1 * 100
            print(f"{noise:<15.2f} {q1:<20} {q2:<20} {improvement:>13.1f}%")
        else:
            print(f"{noise:<15.2f} {q1:<20} {q2:<20} {'N/A':>13}")

    print("=" * 80)


def save_robustness_results(module1_results, module2_results, output_dir='./results'):
    """Save robustness results to JSON"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = {
        'baseline_robustness': module1_results,
        'module2_robustness': module2_results
    }

    output_path = Path(output_dir) / 'robustness_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Robustness results saved to: {output_path}")


def main(architecture, seed):
    """Main entry point"""
    print("\n" + "="*80)
    print("Robustness Testing Experiment")
    print("="*80)
    print(f"Architecture: {architecture}")
    print(f"Random Seed: {seed}\n")

    model_path = Path(f"models/{seed}_{architecture}.npz")
    if not model_path.exists():
        print(f"✗ Model file not found: {model_path}")
        print("  Please train the model using: python train_models.py")
        sys.exit(1)

    print(f"✓ Model file found: {model_path}\n")

    noise_levels = [0, 0.01, 0.05, 0.1]

    print("\n" + "#"*80)
    print("# Testing Module 1 Robustness")
    print("#"*80)
    module1_results = test_module1_robustness(architecture, seed, noise_levels)

    print("\n" + "#"*80)
    print("# Testing Module 2 Robustness")
    print("#"*80)
    module2_results = test_module2_robustness(architecture, seed, noise_levels)

    print_robustness_comparison(module1_results, module2_results)

    save_robustness_results(module1_results, module2_results)

    print("\n" + "="*80)
    print("✓ Robustness Testing Completed")
    print("="*80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python robustness_test.py <architecture> [seed]")
        print("Example: python robustness_test.py 10-10-1 42")
        sys.exit(1)

    architecture = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    try:
        main(architecture, seed)
    except KeyboardInterrupt:
        print("\n\n⊘ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
