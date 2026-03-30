"""
Architecture Experiment for 1D Heat Equation PINN
==================================================
Comparing the impact of different network architectures:

Model A: [2, 20, 20, 1]           - Shallow (2 hidden layers, 20 neurons each)
Model B: [2, 50, 50, 50, 1]       - Medium (3 hidden layers, 50 neurons - baseline)
Model C: [2, 30, 30, 30, 30, 30, 1] - Deep (5 hidden layers, 30 neurons each)
Model D: [2, 100, 100, 1]         - Wide (2 hidden layers, 100 neurons each)

All models use the same loss weights: [1, 1, 1, 1] and learning rate: 0.001
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
    Create comparison plots for all architecture configurations.

    Args:
        results: Dictionary containing results for each model
        save_dir: Directory to save comparison plots
    """
    os.makedirs(save_dir, exist_ok=True)

    models = list(results.keys())
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6']

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

    plt.suptitle('Architecture Experiment - Training Loss Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 2. ARCHITECTURE VISUALIZATION
    # =========================================================================
    fig, ax = plt.subplots(figsize=(16, 8))

    # Calculate total parameters for each model
    param_counts = []
    for model_name, data in results.items():
        layers = data['solver'].layers
        params = sum(layers[i] * layers[i+1] + layers[i+1] for i in range(len(layers)-1))
        param_counts.append(params)

    # Bar chart of parameters
    x_pos = np.arange(len(models))
    bar_width = 0.35

    bars1 = ax.bar(x_pos - bar_width/2, param_counts, bar_width, label='Parameters', color='steelblue')

    # Training time
    ax2 = ax.twinx()
    train_times = [results[m]['metrics']['training_time'] for m in models]
    bars2 = ax2.bar(x_pos + bar_width/2, train_times, bar_width, label='Training Time', color='coral')

    ax.set_xlabel('Model Architecture')
    ax.set_ylabel('Number of Parameters', color='steelblue')
    ax2.set_ylabel('Training Time (s)', color='coral')
    ax.set_xticks(x_pos)

    # Create labels with architecture info
    labels = []
    for model_name, data in results.items():
        arch = data['config']['layers']
        labels.append(f"{model_name}\n{arch}")

    ax.set_xticklabels(labels, fontsize=9)

    # Add value labels
    for bar, val in zip(bars1, param_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val}', ha='center', va='bottom', fontsize=10)

    for bar, val in zip(bars2, train_times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val:.1f}s', ha='center', va='bottom', fontsize=10)

    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title('Architecture Experiment - Model Complexity Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'architecture_comparison.png'), dpi=300, bbox_inches='tight')
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
                axes[i, j].set_ylabel(f'{model_name}\nu(t, x)')

            if i == 0:
                axes[i, j].set_title(f't = {t:.2f}')

            axes[i, j].legend(loc='upper right', fontsize=8)
            axes[i, j].grid(True, alpha=0.3)
            axes[i, j].set_ylim([-0.1, 1.1])

    plt.suptitle('Architecture Experiment - Solution Snapshots Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'snapshots_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 4. METRICS SUMMARY TABLE
    # =========================================================================
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('off')

    table_data = []
    headers = ['Model', 'Architecture', 'Parameters', 'Train Time\n(s)', 'Total Loss',
               'Data Loss', 'BC Loss', 'IC Loss', 'PDE Loss']

    for model_name, data in results.items():
        m = data['metrics']
        layers = data['config']['layers']
        params = sum(layers[i] * layers[i+1] + layers[i+1] for i in range(len(layers)-1))

        row = [
            model_name,
            str(layers),
            str(params),
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
        colColours=['#e74c3c']*len(headers)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    for i in range(len(headers)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    plt.title('Architecture Experiment - Metrics Summary', fontsize=16, fontweight='bold', pad=20)
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
        axes[0, i].set_title(f'{model_name}')
        plt.colorbar(im0, ax=axes[0, i])

        im1 = axes[1, i].pcolormesh(T, X, error, cmap='viridis', shading='auto')
        axes[1, i].set_xlabel('Time (t)')
        axes[1, i].set_ylabel('Position (x)')
        axes[1, i].set_title(f'Error (Max: {error.max():.2e})')
        plt.colorbar(im1, ax=axes[1, i])

    axes[0, 0].set_ylabel('Solution\nPosition (x)')
    axes[1, 0].set_ylabel('Error\nPosition (x)')

    plt.suptitle('Architecture Experiment - Heatmap Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'heatmap_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 6. CONVERGENCE ANALYSIS
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss vs Parameters
    params_list = []
    final_losses = []
    for model_name, data in results.items():
        layers = data['config']['layers']
        params = sum(layers[i] * layers[i+1] + layers[i+1] for i in range(len(layers)-1))
        params_list.append(params)
        final_losses.append(data['metrics']['final_total_loss'])

    axes[0].scatter(params_list, final_losses, s=200, c=colors, edgecolors='black', linewidths=2)
    for i, model_name in enumerate(models):
        axes[0].annotate(model_name, (params_list[i], final_losses[i]),
                         textcoords="offset points", xytext=(0, 10), ha='center')
    axes[0].set_xlabel('Number of Parameters')
    axes[0].set_ylabel('Final Total Loss')
    axes[0].set_title('Loss vs Model Complexity')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    # Loss vs Training Time
    train_times = [results[m]['metrics']['training_time'] for m in models]
    axes[1].scatter(train_times, final_losses, s=200, c=colors, edgecolors='black', linewidths=2)
    for i, model_name in enumerate(models):
        axes[1].annotate(model_name, (train_times[i], final_losses[i]),
                         textcoords="offset points", xytext=(0, 10), ha='center')
    axes[1].set_xlabel('Training Time (s)')
    axes[1].set_ylabel('Final Total Loss')
    axes[1].set_title('Loss vs Training Time')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Architecture Experiment - Efficiency Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'efficiency_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nComparison plots saved to: {save_dir}")


def main():
    """Run architecture experiment with 4 different configurations."""

    # Set random seed for reproducibility
    set_seed(42)

    # ==========================================================================
    # EXPERIMENT CONFIGURATIONS
    # ==========================================================================
    experiments = {
        'Model_A_Shallow': {
            'layers': [2, 20, 20, 1],
            'description': 'Shallow (2 hidden, 20 neurons)'
        },
        'Model_B_Medium': {
            'layers': [2, 50, 50, 50, 1],
            'description': 'Medium (3 hidden, 50 neurons)'
        },
        'Model_C_Deep': {
            'layers': [2, 30, 30, 30, 30, 30, 1],
            'description': 'Deep (5 hidden, 30 neurons)'
        },
        'Model_D_Wide': {
            'layers': [2, 100, 100, 1],
            'description': 'Wide (2 hidden, 100 neurons)'
        }
    }

    # Common parameters
    common_config = {
        'activation': 'tanh',
        'loss_weights': [1.0, 1.0, 1.0, 1.0],
        'learning_rate': 0.001,
        'epochs': 2000,
        'c': 0.1
    }

    # Base directory
    base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'results',
        'architecture'
    )

    # ==========================================================================
    # RUN EXPERIMENTS
    # ==========================================================================
    print("\n" + "="*70)
    print("ARCHITECTURE EXPERIMENT FOR 1D HEAT EQUATION PINN")
    print("="*70)
    print("\nExperiment Overview:")
    print(f"  Activation: {common_config['activation']}")
    print(f"  Loss Weights: {common_config['loss_weights']}")
    print(f"  Learning Rate: {common_config['learning_rate']}")
    print(f"  Epochs: {common_config['epochs']}")
    print("\nArchitecture Configurations:")
    for name, config in experiments.items():
        layers = config['layers']
        params = sum(layers[i] * layers[i+1] + layers[i+1] for i in range(len(layers)-1))
        print(f"  {name}: {layers} - {config['description']} ({params} params)")
    print("="*70)

    results = {}

    for exp_name, exp_config in experiments.items():
        # Reset seed for each experiment for fair comparison
        set_seed(42)

        # Create solver
        solver = HeatEquationPINN(
            layers=exp_config['layers'],
            activation=common_config['activation'],
            loss_weights=common_config['loss_weights'],
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
    print("ARCHITECTURE EXPERIMENT SUMMARY")
    print("="*70)
    print(f"\n{'Model':<25} {'Architecture':<30} {'Params':<10} {'Time (s)':<12} {'Final Loss':<15}")
    print("-"*90)

    for exp_name, data in results.items():
        arch = str(data['config']['layers'])
        layers = data['config']['layers']
        params = sum(layers[i] * layers[i+1] + layers[i+1] for i in range(len(layers)-1))
        time_s = data['metrics']['training_time']
        loss = data['metrics']['final_total_loss']
        print(f"{exp_name:<25} {arch:<30} {params:<10} {time_s:<12.2f} {loss:<15.6e}")

    print("="*90)

    # Find best model
    best_model = min(results.keys(), key=lambda k: results[k]['metrics']['final_total_loss'])
    print(f"\nBest performing model: {best_model}")
    print(f"  Architecture: {results[best_model]['config']['layers']}")
    print(f"  Final Loss: {results[best_model]['metrics']['final_total_loss']:.6e}")

    # Save summary
    summary = {
        'experiment': 'architecture',
        'models': {},
        'best_model': best_model
    }
    for name, data in results.items():
        layers = data['config']['layers']
        params = sum(layers[i] * layers[i+1] + layers[i+1] for i in range(len(layers)-1))
        summary['models'][name] = {
            'layers': layers,
            'parameters': params,
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
