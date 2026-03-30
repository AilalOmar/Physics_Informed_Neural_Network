"""
Learning Rate Experiment for 1D Heat Equation PINN
===================================================
Comparing the impact of different learning rates:

Model A: LR = 0.01   - High learning rate
Model B: LR = 0.001  - Medium learning rate (baseline)
Model C: LR = 0.0001 - Low learning rate

All models use the same architecture: [2, 50, 50, 50, 1]
Loss weights: [1, 1, 1, 1]
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
    Create comparison plots for all learning rate configurations.

    Args:
        results: Dictionary containing results for each model
        save_dir: Directory to save comparison plots
    """
    os.makedirs(save_dir, exist_ok=True)

    models = list(results.keys())
    colors = ['#e74c3c', '#2ecc71', '#3498db']

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

    for bar, val in zip(bars, final_losses):
        axes[1, 2].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            f'{val:.2e}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    plt.suptitle('Learning Rate Experiment - Training Loss Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 2. CONVERGENCE SPEED ANALYSIS
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Early training behavior (first 200 epochs)
    for i, (model_name, data) in enumerate(results.items()):
        epochs = range(1, min(201, len(data['history']['total_loss']) + 1))
        axes[0].semilogy(
            epochs,
            data['history']['total_loss'][:200],
            color=colors[i],
            label=f"{model_name} (LR={data['solver'].learning_rate})",
            linewidth=2
        )
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Early Training (Epochs 1-200)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Late training behavior (last 500 epochs)
    for i, (model_name, data) in enumerate(results.items()):
        n_epochs = len(data['history']['total_loss'])
        start = max(0, n_epochs - 500)
        epochs = range(start + 1, n_epochs + 1)
        axes[1].semilogy(
            epochs,
            data['history']['total_loss'][start:],
            color=colors[i],
            label=f"{model_name} (LR={data['solver'].learning_rate})",
            linewidth=2
        )
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Total Loss')
    axes[1].set_title('Late Training (Last 500 Epochs)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Loss reduction rate
    lr_values = [results[m]['solver'].learning_rate for m in models]
    initial_losses = [results[m]['history']['total_loss'][0] for m in models]
    final_losses = [results[m]['metrics']['final_total_loss'] for m in models]
    reduction_ratios = [init/final for init, final in zip(initial_losses, final_losses)]

    x_pos = np.arange(len(models))
    axes[2].bar(x_pos, reduction_ratios, color=colors)
    axes[2].set_xlabel('Model')
    axes[2].set_ylabel('Loss Reduction Ratio (Initial/Final)')
    axes[2].set_title('Training Efficiency')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels([f"LR={lr}" for lr in lr_values])
    axes[2].set_yscale('log')

    for i, val in enumerate(reduction_ratios):
        axes[2].text(i, val, f'{val:.1e}', ha='center', va='bottom', fontsize=10)

    plt.suptitle('Learning Rate Experiment - Convergence Analysis', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'convergence_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 3. SOLUTION SNAPSHOTS COMPARISON
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
                axes[i, j].set_ylabel(f'LR={data["solver"].learning_rate}\nu(t, x)')

            if i == 0:
                axes[i, j].set_title(f't = {t:.2f}')

            axes[i, j].legend(loc='upper right', fontsize=8)
            axes[i, j].grid(True, alpha=0.3)
            axes[i, j].set_ylim([-0.1, 1.1])

    plt.suptitle('Learning Rate Experiment - Solution Snapshots Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'snapshots_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 4. METRICS SUMMARY TABLE
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    table_data = []
    headers = ['Model', 'Learning Rate', 'Train Time\n(s)', 'Initial Loss',
               'Final Loss', 'Reduction\nRatio', 'BC Loss', 'IC Loss', 'PDE Loss']

    for model_name, data in results.items():
        m = data['metrics']
        init_loss = data['history']['total_loss'][0]
        reduction = init_loss / m['final_total_loss']

        row = [
            model_name,
            f"{data['solver'].learning_rate}",
            f"{m['training_time']:.2f}",
            f"{init_loss:.2e}",
            f"{m['final_total_loss']:.2e}",
            f"{reduction:.2e}",
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
        colColours=['#9b59b6']*len(headers)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    for i in range(len(headers)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    plt.title('Learning Rate Experiment - Metrics Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(save_dir, 'metrics_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 5. HEATMAP COMPARISON
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

        im0 = axes[0, i].pcolormesh(T, X, u_pred, cmap='hot', shading='auto', vmin=0, vmax=1)
        axes[0, i].set_xlabel('Time (t)')
        axes[0, i].set_ylabel('Position (x)')
        axes[0, i].set_title(f'LR = {data["solver"].learning_rate}')
        plt.colorbar(im0, ax=axes[0, i])

        im1 = axes[1, i].pcolormesh(T, X, error, cmap='viridis', shading='auto')
        axes[1, i].set_xlabel('Time (t)')
        axes[1, i].set_ylabel('Position (x)')
        axes[1, i].set_title(f'Error (Max: {error.max():.2e})')
        plt.colorbar(im1, ax=axes[1, i])

    axes[0, 0].set_ylabel('Solution\nPosition (x)')
    axes[1, 0].set_ylabel('Error\nPosition (x)')

    plt.suptitle('Learning Rate Experiment - Heatmap Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'heatmap_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 6. LEARNING RATE VS FINAL LOSS
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    lr_values = [results[m]['solver'].learning_rate for m in models]
    final_losses = [results[m]['metrics']['final_total_loss'] for m in models]
    train_times = [results[m]['metrics']['training_time'] for m in models]

    # LR vs Final Loss
    axes[0].scatter(lr_values, final_losses, s=200, c=colors, edgecolors='black', linewidths=2)
    axes[0].plot(lr_values, final_losses, 'k--', alpha=0.5)
    for i, model_name in enumerate(models):
        axes[0].annotate(model_name, (lr_values[i], final_losses[i]),
                         textcoords="offset points", xytext=(0, 15), ha='center')
    axes[0].set_xlabel('Learning Rate')
    axes[0].set_ylabel('Final Total Loss')
    axes[0].set_title('Learning Rate vs Final Loss')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    # LR vs Training Time
    axes[1].scatter(lr_values, train_times, s=200, c=colors, edgecolors='black', linewidths=2)
    axes[1].plot(lr_values, train_times, 'k--', alpha=0.5)
    for i, model_name in enumerate(models):
        axes[1].annotate(model_name, (lr_values[i], train_times[i]),
                         textcoords="offset points", xytext=(0, 10), ha='center')
    axes[1].set_xlabel('Learning Rate')
    axes[1].set_ylabel('Training Time (s)')
    axes[1].set_title('Learning Rate vs Training Time')
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Learning Rate Experiment - Parameter Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lr_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nComparison plots saved to: {save_dir}")


def main():
    """Run learning rate experiment with 3 different configurations."""

    # Set random seed for reproducibility
    set_seed(42)

    # ==========================================================================
    # EXPERIMENT CONFIGURATIONS
    # ==========================================================================
    experiments = {
        'Model_A_High_LR': {
            'learning_rate': 0.01,
            'description': 'High learning rate'
        },
        'Model_B_Medium_LR': {
            'learning_rate': 0.001,
            'description': 'Medium learning rate (baseline)'
        },
        'Model_C_Low_LR': {
            'learning_rate': 0.0001,
            'description': 'Low learning rate'
        }
    }

    # Common parameters
    common_config = {
        'layers': [2, 50, 50, 50, 1],
        'activation': 'tanh',
        'loss_weights': [1.0, 1.0, 1.0, 1.0],
        'epochs': 2000,
        'c': 0.1
    }

    # Base directory
    base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'results',
        'learning_rate'
    )

    # ==========================================================================
    # RUN EXPERIMENTS
    # ==========================================================================
    print("\n" + "="*70)
    print("LEARNING RATE EXPERIMENT FOR 1D HEAT EQUATION PINN")
    print("="*70)
    print("\nExperiment Overview:")
    print(f"  Architecture: {common_config['layers']}")
    print(f"  Activation: {common_config['activation']}")
    print(f"  Loss Weights: {common_config['loss_weights']}")
    print(f"  Epochs: {common_config['epochs']}")
    print("\nLearning Rate Configurations:")
    for name, config in experiments.items():
        print(f"  {name}: LR = {config['learning_rate']} - {config['description']}")
    print("="*70)

    results = {}

    for exp_name, exp_config in experiments.items():
        # Reset seed for each experiment for fair comparison
        set_seed(42)

        # Create solver
        solver = HeatEquationPINN(
            layers=common_config['layers'],
            activation=common_config['activation'],
            loss_weights=common_config['loss_weights'],
            learning_rate=exp_config['learning_rate'],
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
    print("LEARNING RATE EXPERIMENT SUMMARY")
    print("="*70)
    print(f"\n{'Model':<25} {'Learning Rate':<15} {'Time (s)':<12} {'Final Loss':<15}")
    print("-"*70)

    for exp_name, data in results.items():
        lr = data['solver'].learning_rate
        time_s = data['metrics']['training_time']
        loss = data['metrics']['final_total_loss']
        print(f"{exp_name:<25} {lr:<15} {time_s:<12.2f} {loss:<15.6e}")

    print("="*70)

    # Find best model
    best_model = min(results.keys(), key=lambda k: results[k]['metrics']['final_total_loss'])
    print(f"\nBest performing model: {best_model}")
    print(f"  Learning Rate: {results[best_model]['solver'].learning_rate}")
    print(f"  Final Loss: {results[best_model]['metrics']['final_total_loss']:.6e}")

    # Save summary
    summary = {
        'experiment': 'learning_rate',
        'models': {},
        'best_model': best_model
    }
    for name, data in results.items():
        summary['models'][name] = {
            'learning_rate': data['solver'].learning_rate,
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
