"""
Visualization script: generate comparison plots for Module 1 vs Module 2
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_comparison_results(results_dir='./results'):
    """Load comparison results"""
    possible_paths = [
        Path(results_dir) / 'comparison_results.json',
        Path(results_dir) / 'comparison' / 'comparison_results.json',
        Path(results_dir) / 'real_comparison_results.json',
    ]
    
    for results_path in possible_paths:
        if results_path.exists():
            print(f"Loading comparison results from: {results_path}")
            with open(results_path, 'r') as f:
                return json.load(f)
    
    print(f"Comparison result file not found. Tried:")
    for p in possible_paths:
        print(f"  - {p}")
    return None


def load_robustness_results(results_dir='./results'):
    """Load robustness test results"""
    possible_paths = [
        Path(results_dir) / 'robustness_results.json',
        Path(results_dir) / 'robustness' / 'robustness_results.json',
    ]
    
    for results_path in possible_paths:
        if results_path.exists():
            print(f"Loading robustness results from: {results_path}")
            with open(results_path, 'r') as f:
                return json.load(f)
    
    print(f"Robustness result file not found. Tried:")
    for p in possible_paths:
        print(f"  - {p}")
    return None


def plot_basic_comparison(results, output_dir='./results'):
    """Draw basic comparison charts"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    baseline = results['module1']
    module2 = results['module2']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Module 1 vs Module 2 Comparison', fontsize=16, fontweight='bold')

    # Query count comparison
    ax1 = axes[0, 0]
    methods = ['Module 1\n(Random Sampling)', 'Module 2\n(Structured)']
    queries = [baseline['total_queries'], module2['total_queries']]
    colors = ['#e74c3c', '#3498db']
    bars1 = ax1.bar(methods, queries, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Total Queries', fontsize=12)
    ax1.set_title('Query Count Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    if baseline['total_queries'] > 0:
        improvement = (baseline['total_queries'] - module2['total_queries']) / baseline['total_queries'] * 100
        color = 'green' if improvement > 0 else 'red'
        ax1.text(0.5, max(queries) * 0.5, 
                 f'{"Improvement" if improvement > 0 else "Degradation"}: {abs(improvement):.1f}%',
                 ha='center', fontsize=12,
                 bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    # Time comparison
    ax2 = axes[0, 1]
    times = [baseline['total_time'], module2['total_time']]
    bars2 = ax2.bar(methods, times, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Total Time (seconds)', fontsize=12)
    ax2.set_title('Time Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}s',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    if baseline['total_time'] > 0:
        time_change = (baseline['total_time'] - module2['total_time']) / baseline['total_time'] * 100
        if time_change > 0:
            label = f'Faster: {time_change:.1f}%'
            color_box = 'green'
        else:
            label = f'Slower: {abs(time_change):.1f}%'
            color_box = 'red'
        ax2.text(0.5, max(times) * 0.5, label,
                 ha='center', fontsize=12,
                 bbox=dict(boxstyle='round', facecolor=color_box, alpha=0.3))

    # Iteration comparison
    ax3 = axes[1, 0]
    iter_baseline = baseline.get('iterations', 0)
    iter_module2 = module2.get('iterations', 0)
    iterations = [iter_baseline, iter_module2]

    bars3 = ax3.bar(methods, iterations, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Iterations', fontsize=12)
    ax3.set_title('Iteration Comparison', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    if iter_baseline > 0:
        iter_change = (iter_baseline - iter_module2) / iter_baseline * 100
        if iter_change > 0:
            label = f'Fewer: {iter_change:.1f}%'
            color_box = 'green'
        else:
            label = f'More: {abs(iter_change):.1f}%'
            color_box = 'red'
        ax3.text(0.5, max(iterations) * 0.5, label,
                 ha='center', fontsize=12,
                 bbox=dict(boxstyle='round', facecolor=color_box, alpha=0.3))

    # Efficiency (queries per neuron)
    ax4 = axes[1, 1]
    baseline_efficiency = baseline['total_queries'] / baseline['neurons_recovered'] if baseline['neurons_recovered'] > 0 else 0
    module2_efficiency = module2['total_queries'] / module2['neurons_recovered'] if module2['neurons_recovered'] > 0 else 0
    efficiencies = [baseline_efficiency, module2_efficiency]

    bars4 = ax4.bar(methods, efficiencies, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Queries per Neuron', fontsize=12)
    ax4.set_title('Efficiency Comparison (lower is better)', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    if baseline_efficiency > 0:
        eff_improvement = (baseline_efficiency - module2_efficiency) / baseline_efficiency * 100
        color_box = 'green' if eff_improvement > 0 else 'red'
        ax4.text(0.5, max(efficiencies) * 0.5,
                 f'{"Better" if eff_improvement > 0 else "Worse"}: {abs(eff_improvement):.1f}%',
                 ha='center', fontsize=12,
                 bbox=dict(boxstyle='round', facecolor=color_box, alpha=0.3))

    plt.tight_layout()
    output_path = Path(output_dir) / 'comparison_plots.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plots saved to: {output_path}")
    plt.close()


def plot_robustness_comparison(robustness_results, output_dir='./results'):
    """Draw robustness comparison charts"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    baseline_rob = robustness_results['baseline_robustness']
    module2_rob = robustness_results['module2_robustness']

    noise_levels = [r['noise_level'] for r in baseline_rob]
    baseline_queries = [r['queries'] for r in baseline_rob]
    module2_queries = [r['queries'] for r in module2_rob]
    baseline_success = [100 if r['success'] else 0 for r in baseline_rob]
    module2_success = [100 if r['success'] else 0 for r in module2_rob]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Robustness Test Results', fontsize=16, fontweight='bold')

    # Success rate
    ax1.plot(noise_levels, baseline_success, 'o-', label='Module 1',
             color='#e74c3c', linewidth=2, markersize=8)
    ax1.plot(noise_levels, module2_success, 's-', label='Module 2',
             color='#3498db', linewidth=2, markersize=8)
    ax1.set_xlabel('Noise Level', fontsize=12)
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.set_title('Success Rate vs Noise Level', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-5, 105])

    # Query count vs noise
    ax2.plot(noise_levels, baseline_queries, 'o-', label='Module 1',
             color='#e74c3c', linewidth=2, markersize=8)
    ax2.plot(noise_levels, module2_queries, 's-', label='Module 2',
             color='#3498db', linewidth=2, markersize=8)
    ax2.set_xlabel('Noise Level', fontsize=12)
    ax2.set_ylabel('Queries', fontsize=12)
    ax2.set_title('Queries vs Noise Level', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / 'robustness_plots.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Robustness plots saved to: {output_path}")
    plt.close()


def plot_improvement_summary(results, output_dir='./results'):
    """Draw improvement summary chart with proper handling of negative changes"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    improvements = results.get('improvements', {})

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['Queries\n(fewer is better)', 'Time\n(faster is better)', 'Iterations\n(fewer is better)']
    values = [
        improvements.get('queries', 0),
        improvements.get('time', 0),
        improvements.get('iterations', 0)
    ]

    # Positive = improvement (green), Negative = degradation (red)
    colors_map = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]

    bars = ax.barh(metrics, values, color=colors_map, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Change (%)', fontsize=12)
    ax.set_title('Module 2 vs Module 1 Performance Change', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    ax.grid(axis='x', alpha=0.3)

    # Add annotations
    for bar, value in zip(bars, values):
        width = bar.get_width()
        label = f'{value:+.1f}%'
        if value > 0:
            label = f'✓ {label}'
        else:
            label = f'✗ {label}'
        
        ax.text(width, bar.get_y() + bar.get_height()/2., label,
                ha='left' if value > 0 else 'right',
                va='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Add legend
    ax.text(0.02, 0.98, '✓ = Module 2 better\n✗ = Module 1 better',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = Path(output_dir) / 'improvement_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Improvement summary saved to: {output_path}")
    plt.close()


def main():
    """Main function"""
    print("\n" + "="*70)
    print("Generating visualization plots")
    print("="*70)

    results_dir = './results'
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    comparison_results = load_comparison_results(results_dir)
    if comparison_results:
        print("\n✓ Comparison results loaded successfully")
        print("\nGenerating basic comparison plots...")
        plot_basic_comparison(comparison_results, results_dir)

        print("\nGenerating improvement summary plot...")
        plot_improvement_summary(comparison_results, results_dir)
    else:
        print("\n✗ Comparison results not found, skipping comparison plots")

    robustness_results = load_robustness_results(results_dir)
    if robustness_results:
        print("\n✓ Robustness results loaded successfully")
        print("\nGenerating robustness plots...")
        plot_robustness_comparison(robustness_results, results_dir)
    else:
        print("\n✗ Robustness results not found, skipping robustness plots")

    print("\n" + "="*70)
    print("Visualization complete!")
    print("="*70)
    print(f"\nCheck results in: {Path(results_dir).absolute()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()