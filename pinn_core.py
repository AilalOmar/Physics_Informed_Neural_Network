"""
PINN Core Module for 1D Heat Equation
======================================
Physics-Informed Neural Network implementation for solving:
    ∂u/∂t = c² ∂²u/∂x²

Domain: x ∈ [0, 1], t ∈ [0, 1]
Heat coefficient: c = 0.1
Initial condition: u(0, x) = sin(πx)
Boundary conditions: u(t, 0) = 0, u(t, 1) = 0

Author: PINN Heat Equation Project
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os
from typing import List, Dict, Tuple, Optional


# =============================================================================
# REPRODUCIBILITY SETTINGS
# =============================================================================
def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# NEURAL NETWORK ARCHITECTURE
# =============================================================================
class PINN(nn.Module):
    """
    Physics-Informed Neural Network for solving PDEs.

    Attributes:
        layers: List defining network architecture [input_dim, hidden1, ..., output_dim]
        activation: Activation function (default: tanh)
    """

    def __init__(self, layers: List[int], activation: str = 'tanh'):
        """
        Initialize the PINN model.

        Args:
            layers: List of integers defining layer sizes [2, 50, 50, 50, 1]
            activation: Activation function name ('tanh', 'relu', 'sigmoid')
        """
        super(PINN, self).__init__()

        self.layers = layers
        self.activation_name = activation

        # Select activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Tanh()

        # Build network layers
        self.linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.linears.append(nn.Linear(layers[i], layers[i+1]))

        # Initialize weights using Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (N, 2) containing [t, x] pairs

        Returns:
            Output tensor of shape (N, 1) containing u(t, x) predictions
        """
        for i, linear in enumerate(self.linears[:-1]):
            x = self.activation(linear(x))
        x = self.linears[-1](x)  # No activation on output layer
        return x


# =============================================================================
# HEAT EQUATION SOLVER
# =============================================================================
class HeatEquationPINN:
    """
    PINN solver for the 1D Heat Equation.

    Solves: ∂u/∂t = c² ∂²u/∂x²
    with specified initial and boundary conditions.
    """

    def __init__(
        self,
        layers: List[int] = [2, 50, 50, 50, 1],
        activation: str = 'tanh',
        loss_weights: List[float] = [1.0, 1.0, 1.0, 1.0],
        learning_rate: float = 0.001,
        c: float = 0.1,
        device: str = None
    ):
        """
        Initialize the Heat Equation PINN solver.

        Args:
            layers: Network architecture
            activation: Activation function
            loss_weights: Weights for [Data, BC, IC, PDE] losses
            learning_rate: Learning rate for optimizer
            c: Heat diffusion coefficient
            device: Computing device ('cuda' or 'cpu')
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Store parameters
        self.layers = layers
        self.activation = activation
        self.loss_weights = loss_weights
        self.learning_rate = learning_rate
        self.c = c  # Heat coefficient

        # Initialize model
        self.model = PINN(layers, activation).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training history
        self.history = {
            'total_loss': [],
            'data_loss': [],
            'bc_loss': [],
            'ic_loss': [],
            'pde_loss': []
        }

        # Training time
        self.training_time = 0.0

    def analytical_solution(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute analytical solution for comparison.

        The analytical solution for this problem is:
        u(t, x) = sin(πx) * exp(-c²π²t)

        Args:
            t: Time values
            x: Spatial values

        Returns:
            Analytical solution values
        """
        return torch.sin(np.pi * x) * torch.exp(-(self.c**2) * (np.pi**2) * t)

    def generate_training_data(
        self,
        n_data: int = 100,
        n_bc: int = 100,
        n_ic: int = 100,
        n_pde: int = 10000
    ) -> Dict[str, torch.Tensor]:
        """
        Generate training data points.

        Args:
            n_data: Number of data points (from analytical solution)
            n_bc: Number of boundary condition points
            n_ic: Number of initial condition points
            n_pde: Number of PDE collocation points

        Returns:
            Dictionary containing all training data tensors
        """
        # Data points (from analytical solution for validation)
        t_data = torch.rand(n_data, 1)
        x_data = torch.rand(n_data, 1)
        u_data = self.analytical_solution(t_data, x_data)

        # Boundary condition points: u(t, 0) = 0 and u(t, 1) = 0
        t_bc = torch.rand(n_bc, 1)
        x_bc_left = torch.zeros(n_bc // 2, 1)
        x_bc_right = torch.ones(n_bc // 2, 1)
        x_bc = torch.cat([x_bc_left, x_bc_right], dim=0)
        t_bc = torch.cat([t_bc[:n_bc//2], t_bc[n_bc//2:]], dim=0)
        u_bc = torch.zeros(n_bc, 1)

        # Initial condition points: u(0, x) = sin(πx)
        t_ic = torch.zeros(n_ic, 1)
        x_ic = torch.rand(n_ic, 1)
        u_ic = torch.sin(np.pi * x_ic)

        # PDE collocation points (interior domain)
        t_pde = torch.rand(n_pde, 1)
        x_pde = torch.rand(n_pde, 1)

        # Move all tensors to device
        data = {
            't_data': t_data.to(self.device),
            'x_data': x_data.to(self.device),
            'u_data': u_data.to(self.device),
            't_bc': t_bc.to(self.device),
            'x_bc': x_bc.to(self.device),
            'u_bc': u_bc.to(self.device),
            't_ic': t_ic.to(self.device),
            'x_ic': x_ic.to(self.device),
            'u_ic': u_ic.to(self.device),
            't_pde': t_pde.to(self.device),
            'x_pde': x_pde.to(self.device)
        }

        return data

    def compute_pde_residual(
        self,
        t: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute PDE residual: ∂u/∂t - c² ∂²u/∂x²

        Args:
            t: Time tensor (requires_grad=True)
            x: Spatial tensor (requires_grad=True)

        Returns:
            PDE residual tensor
        """
        t.requires_grad_(True)
        x.requires_grad_(True)

        # Combine inputs
        inputs = torch.cat([t, x], dim=1)

        # Forward pass
        u = self.model(inputs)

        # Compute gradients
        # First derivatives
        grad_outputs = torch.ones_like(u)
        grads = torch.autograd.grad(u, [t, x], grad_outputs=grad_outputs, create_graph=True)
        u_t = grads[0]  # ∂u/∂t
        u_x = grads[1]  # ∂u/∂x

        # Second derivative ∂²u/∂x²
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        # PDE residual: ∂u/∂t - c² ∂²u/∂x²
        residual = u_t - (self.c ** 2) * u_xx

        return residual

    def compute_losses(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """
        Compute all loss components.

        Args:
            data: Training data dictionary

        Returns:
            Tuple of (total_loss, data_loss, bc_loss, ic_loss, pde_loss)
        """
        # Data loss (MSE with analytical solution)
        inputs_data = torch.cat([data['t_data'], data['x_data']], dim=1)
        u_pred_data = self.model(inputs_data)
        data_loss = torch.mean((u_pred_data - data['u_data']) ** 2)

        # Boundary condition loss
        inputs_bc = torch.cat([data['t_bc'], data['x_bc']], dim=1)
        u_pred_bc = self.model(inputs_bc)
        bc_loss = torch.mean((u_pred_bc - data['u_bc']) ** 2)

        # Initial condition loss
        inputs_ic = torch.cat([data['t_ic'], data['x_ic']], dim=1)
        u_pred_ic = self.model(inputs_ic)
        ic_loss = torch.mean((u_pred_ic - data['u_ic']) ** 2)

        # PDE residual loss
        pde_residual = self.compute_pde_residual(data['t_pde'], data['x_pde'])
        pde_loss = torch.mean(pde_residual ** 2)

        # Total weighted loss
        total_loss = (
            self.loss_weights[0] * data_loss +
            self.loss_weights[1] * bc_loss +
            self.loss_weights[2] * ic_loss +
            self.loss_weights[3] * pde_loss
        )

        return total_loss, data_loss, bc_loss, ic_loss, pde_loss

    def train(
        self,
        epochs: int = 2000,
        data: Dict[str, torch.Tensor] = None,
        verbose: bool = True,
        print_every: int = 100
    ) -> Dict[str, List[float]]:
        """
        Train the PINN model.

        Args:
            epochs: Number of training epochs
            data: Training data (generated if None)
            verbose: Print training progress
            print_every: Print frequency

        Returns:
            Training history dictionary
        """
        if data is None:
            data = self.generate_training_data()

        start_time = time.time()

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # Compute losses
            total_loss, data_loss, bc_loss, ic_loss, pde_loss = self.compute_losses(data)

            # Backward pass
            total_loss.backward()
            self.optimizer.step()

            # Record history
            self.history['total_loss'].append(total_loss.item())
            self.history['data_loss'].append(data_loss.item())
            self.history['bc_loss'].append(bc_loss.item())
            self.history['ic_loss'].append(ic_loss.item())
            self.history['pde_loss'].append(pde_loss.item())

            # Print progress
            if verbose and (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Total: {total_loss.item():.6f} | "
                      f"Data: {data_loss.item():.6f} | "
                      f"BC: {bc_loss.item():.6f} | "
                      f"IC: {ic_loss.item():.6f} | "
                      f"PDE: {pde_loss.item():.6f}")

        self.training_time = time.time() - start_time

        if verbose:
            print(f"\nTraining completed in {self.training_time:.2f} seconds")
            print(f"Final Total Loss: {self.history['total_loss'][-1]:.6f}")

        return self.history

    def predict(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            t: Time values (numpy array)
            x: Spatial values (numpy array)

        Returns:
            Predicted u values (numpy array)
        """
        self.model.eval()
        with torch.no_grad():
            t_tensor = torch.tensor(t, dtype=torch.float32).to(self.device)
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

            if t_tensor.dim() == 1:
                t_tensor = t_tensor.unsqueeze(1)
            if x_tensor.dim() == 1:
                x_tensor = x_tensor.unsqueeze(1)

            inputs = torch.cat([t_tensor, x_tensor], dim=1)
            u_pred = self.model(inputs)

        return u_pred.cpu().numpy()

    def save_model(self, filepath: str):
        """Save the trained model and metadata."""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'layers': self.layers,
            'activation': self.activation,
            'loss_weights': self.loss_weights,
            'learning_rate': self.learning_rate,
            'c': self.c,
            'history': self.history,
            'training_time': self.training_time
        }
        torch.save(save_dict, filepath)

    def load_model(self, filepath: str):
        """Load a trained model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint['history']
        self.training_time = checkpoint['training_time']


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
class PINNVisualizer:
    """Visualization utilities for PINN results."""

    def __init__(self, save_dir: str = 'results'):
        """
        Initialize visualizer.

        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Set matplotlib style
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['legend.fontsize'] = 10

    def plot_training_loss(
        self,
        history: Dict[str, List[float]],
        title: str = 'Training Loss',
        filename: str = 'training_loss.png'
    ):
        """
        Plot training loss curves.

        Args:
            history: Training history dictionary
            title: Plot title
            filename: Output filename
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        epochs = range(1, len(history['total_loss']) + 1)

        # Total loss
        axes[0, 0].semilogy(epochs, history['total_loss'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].grid(True, alpha=0.3)

        # Data loss
        axes[0, 1].semilogy(epochs, history['data_loss'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Data Loss')
        axes[0, 1].grid(True, alpha=0.3)

        # BC loss
        axes[0, 2].semilogy(epochs, history['bc_loss'], 'r-', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].set_title('Boundary Condition Loss')
        axes[0, 2].grid(True, alpha=0.3)

        # IC loss
        axes[1, 0].semilogy(epochs, history['ic_loss'], 'm-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Initial Condition Loss')
        axes[1, 0].grid(True, alpha=0.3)

        # PDE loss
        axes[1, 1].semilogy(epochs, history['pde_loss'], 'c-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('PDE Residual Loss')
        axes[1, 1].grid(True, alpha=0.3)

        # All losses combined
        axes[1, 2].semilogy(epochs, history['total_loss'], 'b-', label='Total', linewidth=2)
        axes[1, 2].semilogy(epochs, history['data_loss'], 'g--', label='Data', linewidth=1.5)
        axes[1, 2].semilogy(epochs, history['bc_loss'], 'r--', label='BC', linewidth=1.5)
        axes[1, 2].semilogy(epochs, history['ic_loss'], 'm--', label='IC', linewidth=1.5)
        axes[1, 2].semilogy(epochs, history['pde_loss'], 'c--', label='PDE', linewidth=1.5)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].set_title('All Loss Components')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_heatmap(
        self,
        solver: HeatEquationPINN,
        title: str = 'Heat Distribution u(t, x)',
        filename: str = 'heatmap.png',
        n_points: int = 100
    ):
        """
        Plot heat map of the solution over time.

        Args:
            solver: Trained PINN solver
            title: Plot title
            filename: Output filename
            n_points: Number of grid points
        """
        # Create grid
        t_vals = np.linspace(0, 1, n_points)
        x_vals = np.linspace(0, 1, n_points)
        T, X = np.meshgrid(t_vals, x_vals)

        # Flatten for prediction
        t_flat = T.flatten().reshape(-1, 1)
        x_flat = X.flatten().reshape(-1, 1)

        # Predict
        u_pred = solver.predict(t_flat, x_flat).reshape(n_points, n_points)

        # Analytical solution
        t_tensor = torch.tensor(t_flat, dtype=torch.float32)
        x_tensor = torch.tensor(x_flat, dtype=torch.float32)
        u_exact = solver.analytical_solution(t_tensor, x_tensor).numpy().reshape(n_points, n_points)

        # Error
        error = np.abs(u_pred - u_exact)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Predicted solution
        im0 = axes[0].pcolormesh(T, X, u_pred, cmap='hot', shading='auto')
        axes[0].set_xlabel('Time (t)')
        axes[0].set_ylabel('Position (x)')
        axes[0].set_title('PINN Prediction')
        plt.colorbar(im0, ax=axes[0], label='u(t, x)')

        # Exact solution
        im1 = axes[1].pcolormesh(T, X, u_exact, cmap='hot', shading='auto')
        axes[1].set_xlabel('Time (t)')
        axes[1].set_ylabel('Position (x)')
        axes[1].set_title('Analytical Solution')
        plt.colorbar(im1, ax=axes[1], label='u(t, x)')

        # Error
        im2 = axes[2].pcolormesh(T, X, error, cmap='viridis', shading='auto')
        axes[2].set_xlabel('Time (t)')
        axes[2].set_ylabel('Position (x)')
        axes[2].set_title(f'Absolute Error (Max: {error.max():.2e})')
        plt.colorbar(im2, ax=axes[2], label='|Error|')

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_snapshots(
        self,
        solver: HeatEquationPINN,
        time_points: List[float] = [0, 0.25, 0.5, 0.75, 1.0],
        title: str = 'Solution Snapshots',
        filename: str = 'snapshots.png',
        n_points: int = 100
    ):
        """
        Plot solution snapshots at different time points.

        Args:
            solver: Trained PINN solver
            time_points: Time values for snapshots
            title: Plot title
            filename: Output filename
            n_points: Number of spatial points
        """
        fig, axes = plt.subplots(1, len(time_points), figsize=(4*len(time_points), 4))

        x_vals = np.linspace(0, 1, n_points)

        for i, t in enumerate(time_points):
            t_vals = np.full_like(x_vals, t)

            # PINN prediction
            u_pred = solver.predict(t_vals.reshape(-1, 1), x_vals.reshape(-1, 1)).flatten()

            # Analytical solution
            t_tensor = torch.tensor(t_vals.reshape(-1, 1), dtype=torch.float32)
            x_tensor = torch.tensor(x_vals.reshape(-1, 1), dtype=torch.float32)
            u_exact = solver.analytical_solution(t_tensor, x_tensor).numpy().flatten()

            # Plot
            axes[i].plot(x_vals, u_exact, 'b-', label='Exact', linewidth=2)
            axes[i].plot(x_vals, u_pred, 'r--', label='PINN', linewidth=2)
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('u(t, x)')
            axes[i].set_title(f't = {t:.2f}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim([-0.1, 1.1])

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_pde_residual(
        self,
        solver: HeatEquationPINN,
        title: str = 'PDE Residual',
        filename: str = 'pde_residual.png',
        n_points: int = 50
    ):
        """
        Plot PDE residual across the domain.

        Args:
            solver: Trained PINN solver
            title: Plot title
            filename: Output filename
            n_points: Number of grid points
        """
        # Create grid
        t_vals = np.linspace(0.01, 0.99, n_points)  # Avoid boundaries
        x_vals = np.linspace(0.01, 0.99, n_points)
        T, X = np.meshgrid(t_vals, x_vals)

        # Compute residual
        t_flat = torch.tensor(T.flatten().reshape(-1, 1), dtype=torch.float32).to(solver.device)
        x_flat = torch.tensor(X.flatten().reshape(-1, 1), dtype=torch.float32).to(solver.device)

        solver.model.eval()
        residual = solver.compute_pde_residual(t_flat, x_flat)
        residual_np = residual.detach().cpu().numpy().reshape(n_points, n_points)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Residual heatmap
        im0 = axes[0].pcolormesh(T, X, np.abs(residual_np), cmap='hot', shading='auto')
        axes[0].set_xlabel('Time (t)')
        axes[0].set_ylabel('Position (x)')
        axes[0].set_title('|PDE Residual|')
        plt.colorbar(im0, ax=axes[0], label='|Residual|')

        # Residual histogram
        axes[1].hist(np.abs(residual_np).flatten(), bins=50, color='steelblue', edgecolor='black')
        axes[1].set_xlabel('|PDE Residual|')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Residual Distribution (Mean: {np.mean(np.abs(residual_np)):.2e})')
        axes[1].set_yscale('log')

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def save_metrics(
        self,
        solver: HeatEquationPINN,
        model_name: str,
        filename: str = 'metrics.json'
    ):
        """
        Save training metrics to JSON file.

        Args:
            solver: Trained PINN solver
            model_name: Name identifier for the model
            filename: Output filename
        """
        metrics = {
            'model_name': model_name,
            'architecture': solver.layers,
            'activation': solver.activation,
            'loss_weights': solver.loss_weights,
            'learning_rate': solver.learning_rate,
            'training_time': solver.training_time,
            'final_total_loss': solver.history['total_loss'][-1],
            'final_data_loss': solver.history['data_loss'][-1],
            'final_bc_loss': solver.history['bc_loss'][-1],
            'final_ic_loss': solver.history['ic_loss'][-1],
            'final_pde_loss': solver.history['pde_loss'][-1],
            'total_epochs': len(solver.history['total_loss'])
        }

        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4)

        return metrics


def run_experiment(
    solver: HeatEquationPINN,
    model_name: str,
    save_dir: str,
    epochs: int = 2000,
    verbose: bool = True
) -> Dict:
    """
    Run a complete training experiment with visualization.

    Args:
        solver: PINN solver instance
        model_name: Name identifier for the model
        save_dir: Directory to save results
        epochs: Number of training epochs
        verbose: Print training progress

    Returns:
        Dictionary containing metrics and results
    """
    # Initialize visualizer
    viz = PINNVisualizer(save_dir)

    # Train model
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"Architecture: {solver.layers}")
    print(f"Loss Weights: {solver.loss_weights}")
    print(f"Learning Rate: {solver.learning_rate}")
    print(f"{'='*60}\n")

    history = solver.train(epochs=epochs, verbose=verbose)

    # Generate all plots
    viz.plot_training_loss(
        history,
        title=f'{model_name} - Training Loss',
        filename='training_loss.png'
    )

    viz.plot_heatmap(
        solver,
        title=f'{model_name} - Heat Distribution',
        filename='heatmap.png'
    )

    viz.plot_snapshots(
        solver,
        title=f'{model_name} - Solution Snapshots',
        filename='snapshots.png'
    )

    viz.plot_pde_residual(
        solver,
        title=f'{model_name} - PDE Residual',
        filename='pde_residual.png'
    )

    # Save metrics
    metrics = viz.save_metrics(solver, model_name)

    # Save model
    model_path = os.path.join(save_dir, 'model.pt')
    solver.save_model(model_path)

    print(f"\nResults saved to: {save_dir}")
    print(f"Training Time: {solver.training_time:.2f} seconds")
    print(f"Final Loss: {history['total_loss'][-1]:.6f}")

    return metrics
