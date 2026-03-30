"""
Baseline PINN Model for 1D Heat Equation
=========================================
Reference model with standard configuration:
- Architecture: [2, 50, 50, 50, 1]
- Activation: tanh
- Loss weights: [1, 1, 1, 1] for [Data, BC, IC, PDE]
- Learning rate: 0.001
- Training epochs: 2000

This serves as the reference point for all other experiments.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pinn_core import (
    set_seed,
    HeatEquationPINN,
    run_experiment
)


def main():
    """Run the baseline PINN model."""

    # Set random seed for reproducibility
    set_seed(42)

    # ==========================================================================
    # BASELINE CONFIGURATION
    # ==========================================================================
    config = {
        'model_name': 'Baseline',
        'layers': [2, 50, 50, 50, 1],      # Architecture: input -> 3 hidden -> output
        'activation': 'tanh',               # Activation function
        'loss_weights': [1.0, 1.0, 1.0, 1.0],  # [Data, BC, IC, PDE] weights
        'learning_rate': 0.001,             # Learning rate
        'epochs': 2000,                     # Number of training epochs
        'c': 0.1                            # Heat diffusion coefficient
    }

    # Output directory
    save_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'results',
        'baseline'
    )
    os.makedirs(save_dir, exist_ok=True)

    # ==========================================================================
    # CREATE AND TRAIN MODEL
    # ==========================================================================
    print("\n" + "="*70)
    print("BASELINE PINN MODEL FOR 1D HEAT EQUATION")
    print("="*70)
    print(f"\nProblem Setup:")
    print(f"  Domain: x ∈ [0, 1], t ∈ [0, 1]")
    print(f"  Heat coefficient: c = {config['c']}")
    print(f"  Initial condition: u(0, x) = sin(πx)")
    print(f"  Boundary conditions: u(t, 0) = 0, u(t, 1) = 0")
    print(f"\nModel Configuration:")
    print(f"  Architecture: {config['layers']}")
    print(f"  Activation: {config['activation']}")
    print(f"  Loss weights [Data, BC, IC, PDE]: {config['loss_weights']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Epochs: {config['epochs']}")
    print("="*70)

    # Initialize solver
    solver = HeatEquationPINN(
        layers=config['layers'],
        activation=config['activation'],
        loss_weights=config['loss_weights'],
        learning_rate=config['learning_rate'],
        c=config['c']
    )

    # Run experiment
    metrics = run_experiment(
        solver=solver,
        model_name=config['model_name'],
        save_dir=save_dir,
        epochs=config['epochs'],
        verbose=True
    )

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "="*70)
    print("BASELINE MODEL SUMMARY")
    print("="*70)
    print(f"  Training Time:     {metrics['training_time']:.2f} seconds")
    print(f"  Final Total Loss:  {metrics['final_total_loss']:.6e}")
    print(f"  Final Data Loss:   {metrics['final_data_loss']:.6e}")
    print(f"  Final BC Loss:     {metrics['final_bc_loss']:.6e}")
    print(f"  Final IC Loss:     {metrics['final_ic_loss']:.6e}")
    print(f"  Final PDE Loss:    {metrics['final_pde_loss']:.6e}")
    print("="*70)
    print(f"\nAll results saved to: {save_dir}")
    print("Generated files:")
    print("  - training_loss.png")
    print("  - heatmap.png")
    print("  - snapshots.png")
    print("  - pde_residual.png")
    print("  - metrics.json")
    print("  - model.pt")

    return metrics


if __name__ == "__main__":
    metrics = main()
