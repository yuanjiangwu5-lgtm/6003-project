"""
Main script to run all experiments

Includes:
1. Basic comparison experiments
2. Robustness testing
3. Visualization generation
"""

import sys
import subprocess
from pathlib import Path
import time


def run_command(cmd, description):
    """Run a command and show progress"""
    print("\n" + "="*70)
    print(f"Running: {description}")
    print("="*70)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        elapsed = time.time() - start_time
        print(f"\n✓ Completed (time: {elapsed:.1f}s)")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Failed (time: {elapsed:.1f}s)")
        print(f"Error: {e}")
        return False
    except FileNotFoundError:
        print(f"\n✗ File not found: {cmd[0]}")
        return False


def check_model_exists(architecture, seed):
    """Check whether model file exists"""
    model_path = Path(f"models/{seed}_{architecture}.npz")
    if model_path.exists():
        print(f"✓ Model file exists: {model_path}")
        return True
    else:
        print(f"✗ Model file does not exist: {model_path}")
        return False


def main(architecture, seed, skip_training=False, skip_comparison=False, 
         skip_robustness=False, skip_visualization=False):
    """
    Run the full experiment suite
    
    Args:
        architecture: network architecture such as "10-10-1"
        seed: random seed
        skip_training: skip model training
        skip_comparison: skip comparison experiment
        skip_robustness: skip robustness test
        skip_visualization: skip visualization generation
    """
    print("\n" + "="*80)
    print("Running Full Experiment Suite")
    print("="*80)
    print(f"Architecture: {architecture}")
    print(f"Random Seed: {seed}")
    print()
    
    experiment_dir = Path(__file__).parent
    project_root = experiment_dir.parent
    
    results = {
        'training': None,
        'comparison': None,
        'robustness': None,
        'visualization': None
    }
    
    # Step 1: Model training (if needed)
    if not skip_training:
        if not check_model_exists(architecture, seed):
            print("\n" + "#"*80)
            print("# Step 1: Train Model")
            print("#"*80)
            
            cmd = ['python', 'train_models.py', architecture, str(seed)]
            results['training'] = run_command(cmd, "Model Training")
            
            if not results['training']:
                print("\nModel training failed, cannot continue")
                return results
        else:
            print("\nModel already exists, skipping training")
            results['training'] = True
    else:
        print("\nSkipping model training")
        results['training'] = True
    
    # Step 2: Comparison experiment
    if not skip_comparison:
        print("\n" + "#"*80)
        print("# Step 2: Run Module Comparison Experiment")
        print("#"*80)
        
        comparison_paths = [
            experiment_dir / 'fixed_comparison.py',
            project_root / 'experiments' / 'fixed_comparison.py',
            Path('fixed_comparison.py'),
            experiment_dir / 'compare_modules.py',
            project_root / 'experiments' / 'compare_modules.py',
            Path('compare_modules.py'),
        ]
        
        comparison_script = None
        for path in comparison_paths:
            if path.exists():
                comparison_script = str(path)
                break
        
        if comparison_script:
            cmd = ['python', comparison_script, architecture, str(seed)]
            results['comparison'] = run_command(cmd, "Module Comparison Experiment")
        else:
            print(f"✗ Comparison script not found")
            print("Paths attempted:")
            for path in comparison_paths:
                print(f"  - {path}")
            results['comparison'] = False
    else:
        print("\nSkipping comparison experiment")
        results['comparison'] = True
    
    # Step 3: Robustness test
    if not skip_robustness:
        print("\n" + "#"*80)
        print("# Step 3: Run Robustness Test")
        print("#"*80)
        
        robustness_paths = [
            experiment_dir / 'robustness_test.py',
            project_root / 'experiments' / 'robustness_test.py',
            Path('robustness_test.py')
        ]
        
        robustness_script = None
        for path in robustness_paths:
            if path.exists():
                robustness_script = str(path)
                break
        
        if robustness_script:
            cmd = ['python', robustness_script, architecture, str(seed)]
            results['robustness'] = run_command(cmd, "Robustness Test")
        else:
            print(f"✗ Robustness test script not found")
            results['robustness'] = False
    else:
        print("\nSkipping robustness test")
        results['robustness'] = True
    
    # Step 4: Visualization
    if not skip_visualization:
        print("\n" + "#"*80)
        print("# Step 4: Generate Visualizations")
        print("#"*80)
        
        visualization_paths = [
            experiment_dir / 'visualize.py',
            project_root / 'experiments' / 'visualize.py',
            Path('visualize_results.py')
        ]
        
        visualization_script = None
        for path in visualization_paths:
            if path.exists():
                visualization_script = str(path)
                break
        
        if visualization_script:
            cmd = ['python', visualization_script]
            results['visualization'] = run_command(cmd, "Visualization Generation")
        else:
            print(f"✗ Visualization script not found")
            results['visualization'] = False
    else:
        print("\nSkipping visualization generation")
        results['visualization'] = True
    
    # Print summary
    print("\n" + "="*80)
    print("Experiment Suite Summary")
    print("="*80)
    
    for step, result in results.items():
        if result is None:
            status = "Skipped"
        elif result:
            status = "✓ Success"
        else:
            status = "✗ Failed"
        print(f"{step.capitalize():<20} {status}")
    
    all_success = all(r is not False for r in results.values())
    
    if all_success:
        print("\n" + "="*80)
        print("🎉 All Experiments Completed Successfully!")
        print("="*80)
        print("\nResults available at:")
        print("  - results/comparison/comparison_results.json")
        print("  - results/robustness/robustness_results.json")
        print("  - results/figures/*.png")
    else:
        print("\n" + "="*80)
        print("⚠️  Some experiments failed")
        print("="*80)
        print("\nPlease check the errors above")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the full experiment suite')
    parser.add_argument('architecture', type=str, help='Network architecture, e.g. 10-10-1')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    parser.add_argument('--skip-comparison', action='store_true', help='Skip comparison experiment')
    parser.add_argument('--skip-robustness', action='store_true', help='Skip robustness test')
    parser.add_argument('--skip-visualization', action='store_true', help='Skip visualization')
    
    args = parser.parse_args()
    
    try:
        main(
            args.architecture,
            args.seed,
            skip_training=args.skip_training,
            skip_comparison=args.skip_comparison,
            skip_robustness=args.skip_robustness,
            skip_visualization=args.skip_visualization
        )
    except KeyboardInterrupt:
        print("\n\nExperiment aborted by user")
    except Exception as e:
        print(f"\nExperiment suite failed: {e}")
        import traceback
        traceback.print_exc()
