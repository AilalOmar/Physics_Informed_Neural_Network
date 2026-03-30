"""
Run All PINN Experiments
========================
This script runs all PINN experiments sequentially and generates
the final comparison visualizations.

Experiments:
1. Baseline model
2. Loss weights experiment (4 models)
3. Architecture experiment (4 models)
4. Learning rate experiment (3 models)

Total: 12 models trained and compared
"""

import sys
import os
import time
import subprocess
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def run_script(script_name: str, description: str) -> bool:
    """
    Run a Python script and capture output.

    Args:
        script_name: Name of the script to run
        description: Description for logging

    Returns:
        True if successful, False otherwise
    """
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)

    print(f"\n{'='*70}")
    print(f"RUNNING: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*70}\n")

    start_time = time.time()

    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=False,
            text=True
        )

        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✓ {description} completed in {elapsed_time:.2f} seconds")
            return True
        else:
            print(f"\n✗ {description} failed with return code {result.returncode}")
            return False

    except Exception as e:
        print(f"\n✗ Error running {description}: {str(e)}")
        return False


def main():
    """Run all experiments and generate comparison."""

    print("\n" + "="*70)
    print("PINN HEAT EQUATION - COMPLETE EXPERIMENT SUITE")
    print("="*70)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis will run all experiments:")
    print("  1. Baseline model (1 model)")
    print("  2. Loss weights experiment (4 models)")
    print("  3. Architecture experiment (4 models)")
    print("  4. Learning rate experiment (3 models)")
    print("  5. Final comparison analysis")
    print("\nTotal: 12 models to train")
    print("="*70)

    total_start = time.time()

    # Track results
    results = {
        'baseline': False,
        'loss_weights': False,
        'architecture': False,
        'learning_rate': False,
        'comparison': False
    }

    # Run experiments
    experiments = [
        ('baseline.py', 'Baseline Model', 'baseline'),
        ('loss_weights_experiment.py', 'Loss Weights Experiment', 'loss_weights'),
        ('architecture_experiment.py', 'Architecture Experiment', 'architecture'),
        ('learning_rate_experiment.py', 'Learning Rate Experiment', 'learning_rate'),
        ('comparison.py', 'Final Comparison Analysis', 'comparison')
    ]

    for script, description, key in experiments:
        results[key] = run_script(script, description)

        # If comparison failed but experiments succeeded, try again
        if key == 'comparison' and not results[key]:
            print("\nRetrying comparison...")
            results[key] = run_script(script, description)

    total_elapsed = time.time() - total_start

    # Print final summary
    print("\n" + "="*70)
    print("EXPERIMENT SUITE COMPLETE")
    print("="*70)
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {total_elapsed/60:.2f} minutes ({total_elapsed:.2f} seconds)")

    print("\nResults:")
    for key, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {key.replace('_', ' ').title()}: {status}")

    # Output directory
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'results'
    )

    print(f"\nAll results saved to: {results_dir}")
    print("\nDirectory structure:")
    print("  results/")
    print("  ├── baseline/           - Baseline model results")
    print("  ├── loss_weights/       - Loss weights experiment")
    print("  │   ├── Model_A_Equal/")
    print("  │   ├── Model_B_Physics/")
    print("  │   ├── Model_C_Boundary/")
    print("  │   ├── Model_D_Initial/")
    print("  │   └── comparison/")
    print("  ├── architecture/       - Architecture experiment")
    print("  │   ├── Model_A_Shallow/")
    print("  │   ├── Model_B_Medium/")
    print("  │   ├── Model_C_Deep/")
    print("  │   ├── Model_D_Wide/")
    print("  │   └── comparison/")
    print("  ├── learning_rate/      - Learning rate experiment")
    print("  │   ├── Model_A_High_LR/")
    print("  │   ├── Model_B_Medium_LR/")
    print("  │   ├── Model_C_Low_LR/")
    print("  │   └── comparison/")
    print("  └── comparison/         - Final comparison across all experiments")
    print("      ├── 01_all_models_ranking.png")
    print("      ├── 02_efficiency_scatter.png")
    print("      ├── 03_parameter_impact.png")
    print("      ├── 04_loss_components.png")
    print("      ├── 05_comprehensive_summary.png")
    print("      ├── 06_key_insights.png")
    print("      └── final_report.json")

    print("\n" + "="*70)

    # Return success status
    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
