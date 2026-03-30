"""
Loss Weights Experiment for 1D Heat Equation PINN
==================================================
Comparing the impact of different loss weight configurations:

Model A: [1, 1, 1, 1]   - Equal weights (baseline)
Model B: [1, 1, 1, 10]  - Physics-heavy (emphasizes PDE residual)
Model C: [1, 10, 10, 1] - Boundary-heavy (emphasizes BC and IC)
Model D: [1, 1, 10, 1]  - Initial-heavy (emphasizes IC)

All models use the same architecture: [2, 50, 50, 50, 1]
Loss weights format: [Data, BC, IC, PDE]
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pinn_core import (
    set_seed,
    HeatEquationPINN,
    PINNVisualizer,
    run_experiment
)


def create_comparison_plots(results: dict, save_dir: str):
    """
    Create comparison plots for all loss weight configurations.

    Args:
        results: Dictionary containing results for each model
        save_dir: Directory to save comparison plots
    """
    os.makedirs(save_dir, exist_ok=True)

    models = list(results.keys())
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']

    # =========================================================================
    # 1. TRAINING LOSS COMPARISON
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    loss_types = ['total_loss', 'data_loss', 'bc_loss', 'ic_loss', 'pde_loss']
    titles = ['Total Loss', 'Data Loss', 'BC Loss', 'IC Loss', 'PDE Loss']

    for idx, (loss_type, title) in enumerate(zip(loss_types, titles)):
        row, col = idx // 3, idx % 3
        for i, (model_name, data) in enumerate(results.items()):
            epochs = range(1, len(data['history'][loss_type]) + 1)
            axes[row, col].semilogy(
                epochs,
                data['history'][loss_type],
                color=colors[i],
                label=model_name,
                linewidth=2
            )
        axes[row, col].set_xlabel('Epoch')
        axes[row, col].set_ylabel('Loss')
        axes[row, col].set_title(title)
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)

    # Final loss bar chart
    final_losses = [results[m]['metrics']['final_total_loss'] for m in models]
    bars = axes[1, 2].bar(models, final_losses, color=colors)
    axes[1, 2].set_xlabel('Model')
    axes[1, 2].set_ylabel('Final Total Loss')
    axes[1, 2].set_title('Final Loss Comparison')
    axes[1, 2].tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, val in zip(bars, final_losses):
        axes[1, 2].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            f'{val:.2e}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    plt.suptitle('Loss Weights Experiment - Training Loss Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 2. SOLUTION SNAPSHOTS COMPARISON
    # =========================================================================
    time_points = [0, 0.25, 0.5, 0.75, 1.0]
    n_points = 100
    x_vals = np.linspace(0, 1, n_points)

    fig, axes = plt.subplots(len(models), len(time_points), figsize=(20, 4*len(models)))

    for i, (model_name, data) in enumerate(results.items()):
        solver = data['solver']
        for j, t in enumerate(time_points):
            t_vals = np.full_like(x_vals, t)

            # PINN prediction
            u_pred = solver.predict(t_vals.reshape(-1, 1), x_vals.reshape(-1, 1)).flatten()

            # Analytical solution
            import torch
            t_tensor = torch.tensor(t_vals.reshape(-1, 1), dtype=torch.float32)
            x_tensor = torch.tensor(x_vals.reshape(-1, 1), dtype=torch.float32)
            u_exact = solver.analytical_solution(t_tensor, x_tensor).numpy().flatten()

            axes[i, j].plot(x_vals, u_exact, 'b-', label='Exact', linewidth=2)
            axes[i, j].plot(x_vals, u_pred, 'r--', label='PINN', linewidth=2)
            axes[i, j].set_xlabel('x')
            axes[i, j].set_ylabel('u(t, x)')

            if j == 0:
                axes[i, j].set_ylabel(f'{model_name}\nu(t, x)')

            if i == 0:
                axes[i, j].set_title(f't = {t:.2f}')

            axes[i, j].legend(loc='upper right', fontsize=8)
            axes[i, j].grid(True, alpha=0.3)
            axes[i, j].set_ylim([-0.1, 1.1])

    plt.suptitle('Loss Weights Experiment - Solution Snapshots Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'snapshots_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 3. METRICS SUMMARY TABLE
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')

    # Table data
    table_data = []
    headers = ['Model', 'Weights\n[D,BC,IC,PDE]', 'Train Time\n(s)', 'Total Loss',
               'Data Loss', 'BC Loss', 'IC Loss', 'PDE Loss']

    for model_name, data in results.items():
        m = data['metrics']
        row = [
            model_name,
            str(data['solver'].loss_weights),
            f"{m['training_time']:.2f}",
            f"{m['final_total_loss']:.2e}",
            f"{m['final_data_loss']:.2e}",
            f"{m['final_bc_loss']:.2e}",
            f"{m['final_ic_loss']:.2e}",
            f"{m['final_pde_loss']:.2e}"
        ]
        table_data.append(row)

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colColours=['#3498db']*len(headers)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header cells
    for i in range(len(headers)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    plt.title('Loss Weights Experiment - Metrics Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(save_dir, 'metrics_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 4. HEATMAP COMPARISON
    # =========================================================================
    fig, axes = plt.subplots(2, len(models), figsize=(5*len(models), 10))

    for i, (model_name, data) in enumerate(results.items()):
        solver = data['solver']
        n_pts = 100

        t_vals = np.linspace(0, 1, n_pts)
        x_vals = np.linspace(0, 1, n_pts)
        T, X = np.meshgrid(t_vals, x_vals)

        t_flat = T.flatten().reshape(-1, 1)
        x_flat = X.flatten().reshape(-1, 1)

        u_pred = solver.predict(t_flat, x_flat).reshape(n_pts, n_pts)

        import torch
        t_tensor = torch.tensor(t_flat, dtype=torch.float32)
        x_tensor = torch.tensor(x_flat, dtype=torch.float32)
        u_exact = solver.analytical_solution(t_tensor, x_tensor).numpy().reshape(n_pts, n_pts)

        error = np.abs(u_pred - u_exact)

        # Solution heatmap
        im0 = axes[0, i].pcolormesh(T, X, u_pred, cmap='hot', shading='auto', vmin=0, vmax=1)
        axes[0, i].set_xlabel('Time (t)')
        axes[0, i].set_ylabel('Position (x)')
        axes[0, i].set_title(f'{model_name}')
        plt.colorbar(im0, ax=axes[0, i])

        # Error heatmap
        im1 = axes[1, i].pcolormesh(T, X, error, cmap='viridis', shading='auto')
        axes[1, i].set_xlabel('Time (t)')
        axes[1, i].set_ylabel('Position (x)')
        axes[1, i].set_title(f'Error (Max: {error.max():.2e})')
        plt.colorbar(im1, ax=axes[1, i])

    axes[0, 0].set_ylabel('Solution\nPosition (x)')
    axes[1, 0].set_ylabel('Error\nPosition (x)')

    plt.suptitle('Loss Weights Experiment - Heatmap Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'heatmap_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nComparison plots saved to: {save_dir}")


def main():
    """Run loss weights experiment with 4 different configurations."""

    # Set random seed for reproducibility
    set_seed(42)

    # ==========================================================================
    # EXPERIMENT CONFIGURATIONS
    # ==========================================================================
    experiments = {
        'Model_A_Equal': {
            'loss_weights': [1.0, 1.0, 1.0, 1.0],
            'description': 'Equal weights'
        },
        'Model_B_Physics': {
            'loss_weights': [1.0, 1.0, 1.0, 10.0],
            'description': 'Physics-heavy (PDE x10)'
        },
        'Model_C_Boundary': {
            'loss_weights': [1.0, 10.0, 10.0, 1.0],
            'description': 'Boundary-heavy (BC, IC x10)'
        },
        'Model_D_Initial': {
            'loss_weights': [1.0, 1.0, 10.0, 1.0],
            'description': 'Initial-heavy (IC x10)'
        }
    }

    # Common parameters
    common_config = {
        'layers': [2, 50, 50, 50, 1],
        'activation': 'tanh',
        'learning_rate': 0.001,
        'epochs': 2000,
        'c': 0.1
    }

    # Base directory
    base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'results',
        'loss_weights'
    )

    # ==========================================================================
    # RUN EXPERIMENTS
    # ==========================================================================
    print("\n" + "="*70)
    print("LOSS WEIGHTS EXPERIMENT FOR 1D HEAT EQUATION PINN")
    print("="*70)
    print("\nExperiment Overview:")
    print(f"  Common Architecture: {common_config['layers']}")
    print(f"  Activation: {common_config['activation']}")
    print(f"  Learning Rate: {common_config['learning_rate']}")
    print(f"  Epochs: {common_config['epochs']}")
    print("\nWeight Configurations [Data, BC, IC, PDE]:")
    for name, config in experiments.items():
        print(f"  {name}: {config['loss_weights']} - {config['description']}")
    print("="*70)

    results = {}

    for exp_name, exp_config in experiments.items():
        # Reset seed for each experiment for fair comparison
        set_seed(42)

        # Create solver
        solver = HeatEquationPINN(
            layers=common_config['layers'],
            activation=common_config['activation'],
            loss_weights=exp_config['loss_weights'],
            learning_rate=common_config['learning_rate'],
            c=common_config['c']
        )

        # Output directory for this experiment
        save_dir = os.path.join(base_dir, exp_name)

        # Run experiment
        metrics = run_experiment(
            solver=solver,
            model_name=f"{exp_name} ({exp_config['description']})",
            save_dir=save_dir,
            epochs=common_config['epochs'],
            verbose=True
        )

        # Store results
        results[exp_name] = {
            'solver': solver,
            'metrics': metrics,
            'history': solver.history,
            'config': exp_config
        }

    # ==========================================================================
    # CREATE COMPARISON PLOTS
    # ==========================================================================
    comparison_dir = os.path.join(base_dir, 'comparison')
    create_comparison_plots(results, comparison_dir)

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "="*70)
    print("LOSS WEIGHTS EXPERIMENT SUMMARY")
    print("="*70)
    print(f"\n{'Model':<25} {'Weights':<20} {'Time (s)':<12} {'Final Loss':<15}")
    print("-"*70)

    for exp_name, data in results.items():
        weights = str(data['solver'].loss_weights)
        time_s = data['metrics']['training_time']
        loss = data['metrics']['final_total_loss']
        print(f"{exp_name:<25} {weights:<20} {time_s:<12.2f} {loss:<15.6e}")

    print("="*70)

    # Find best model
    best_model = min(results.keys(), key=lambda k: results[k]['metrics']['final_total_loss'])
    print(f"\nBest performing model: {best_model}")
    print(f"  Final Loss: {results[best_model]['metrics']['final_total_loss']:.6e}")

    # Save summary
    summary = {
        'experiment': 'loss_weights',
        'models': {},
        'best_model': best_model
    }
    for name, data in results.items():
        summary['models'][name] = {
            'loss_weights': data['solver'].loss_weights,
            'description': data['config']['description'],
            'training_time': data['metrics']['training_time'],
            'final_total_loss': data['metrics']['final_total_loss'],
            'final_data_loss': data['metrics']['final_data_loss'],
            'final_bc_loss': data['metrics']['final_bc_loss'],
            'final_ic_loss': data['metrics']['final_ic_loss'],
            'final_pde_loss': data['metrics']['final_pde_loss']
        }

    with open(os.path.join(base_dir, 'experiment_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\nAll results saved to: {base_dir}")

    return results


if __name__ == "__main__":
    results = main()
