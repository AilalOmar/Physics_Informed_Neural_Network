# Physics-Informed Neural Networks for 1D Heat Equation

## Project Overview

This project uses **Physics-Informed Neural Networks (PINNs)** to solve the 1D heat equation without relying on traditional numerical methods. Instead of discretizing the domain, we train a neural network that learns the solution while respecting the underlying physics.

## Problem Statement

**Heat Equation:** ∂u/∂t = c² ∂²u/∂x²

- **Domain:** x ∈ [0, 1], t ∈ [0, 1]
- **Heat coefficient:** c = 0.1
- **Initial condition:** u(0, x) = sin(πx)
- **Boundary conditions:** u(t, 0) = 0, u(t, 1) = 0

## Methodology

### Loss Function

The neural network minimizes a composite loss function:

```
L_total = w₁·L_data + w₂·L_BC + w₃·L_IC + w₄·L_PDE
```

Where:
- **L_data**: Data fitting loss (when measurements available)
- **L_BC**: Boundary conditions loss
- **L_IC**: Initial conditions loss
- **L_PDE**: Physics residual (equation must equal zero)

### Experiments Conducted

We compared different configurations to understand their impact on model performance:

#### 1. **Loss Function Weights**
- Equal weights [1, 1, 1, 1]
- Physics-heavy [1, 1, 1, 10]
- Boundary-heavy [1, 10, 10, 1]
- Initial-heavy [1, 1, 10, 1]

#### 2. **Network Architecture**
- Shallow: [2, 20, 20, 1]
- Medium: [2, 50, 50, 50, 1] (baseline)
- Deep: [2, 30, 30, 30, 30, 30, 1]
- Wide: [2, 100, 100, 1]

#### 3. **Learning Rate**
- High: 0.01
- Medium: 0.001 (baseline)
- Low: 0.0001

## Key Findings

- **Architecture matters**: Shallow networks converge stably; deeper networks show periodic oscillations but can achieve better accuracy
- **Loss weighting**: Physics-heavy weights enforce equation satisfaction; boundary-heavy ensures accurate edge conditions
- **Learning rate**: Higher rates train faster but can cause instability in deep networks
- **Trade-offs**: Network complexity vs training stability

## Visualizations

Each experiment produces:
- **Heat maps** showing temperature evolution over time
- **Loss curves** tracking convergence (total + individual components)
- **Solution snapshots** at different time points
- **PDE residual plots** showing where physics is violated

## Project Structure

```
├── baseline.py                    # Reference configuration
├── loss_weights_experiment.py     # Compare loss weighting strategies
├── architecture_experiment.py     # Compare network depths/widths
├── learning_rate_experiment.py    # Compare optimization speeds
├── results/                       # Generated plots and metrics
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib

## Usage

Run individual experiments:
```bash
python baseline.py
python architecture_experiment.py
python loss_weights_experiment.py
python learning_rate_experiment.py
```

Each script trains the models, saves results, and generates comparison plots.

## Conclusion

This project demonstrates that neural networks can solve partial differential equations by embedding physical laws directly into the training process. The choice of architecture, loss weighting, and optimization strategy significantly impacts both solution accuracy and training stability.

---

**Project for:** Scientific Computing / Machine Learning for Physics  
**Topic:** Physics-Informed Neural Networks (PINNs)  
**Equation:** 1D Heat Equation
