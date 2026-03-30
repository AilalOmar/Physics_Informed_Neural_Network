"""
Final Comparison Script for PINN Experiments
=============================================
This script loads all experiment results and generates comprehensive
side-by-side comparisons to analyze the impact of different parameters
on the PINN model's performance for solving the 1D heat equation.

Experiments compared:
1. Baseline model
2. Loss weights variations
3. Architecture variations
4. Learning rate variations
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pinn_core import HeatEquationPINN, set_seed


def load_experiment_results(base_dir: str) -> Dict[str, Any]:
    """
    Load all experiment results from saved files.

    Args:
        base_dir: Base directory containing experiment results

    Returns:
        Dictionary containing all experiment data
    """
    results = {
        'baseline': {},
        'loss_weights': {},
        'architecture': {},
        'learning_rate': {}
    }

    # Load baseline
    baseline_dir = os.path.join(base_dir, 'baseline')
    if os.path.exists(baseline_dir):
        metrics_path = os.path.join(baseline_dir, 'metrics.json')
        model_path = os.path.join(baseline_dir, 'model.pt')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                results['baseline']['metrics'] = json.load(f)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            results['baseline']['history'] = checkpoint['history']
            results['baseline']['config'] = {
                'layers': checkpoint['layers'],
                'loss_weights': checkpoint['loss_weights'],
                'learning_rate': checkpoint['learning_rate']
            }

    # Load loss weights experiment
    lw_dir = os.path.join(base_dir, 'loss_weights')
    lw_summary_path = os.path.join(lw_dir, 'experiment_summary.json')
    if os.path.exists(lw_summary_path):
        with open(lw_summary_path, 'r') as f:
            results['loss_weights'] = json.load(f)

    # Load architecture experiment
    arch_dir = os.path.join(base_dir, 'architecture')
    arch_summary_path = os.path.join(arch_dir, 'experiment_summary.json')
    if os.path.exists(arch_summary_path):
        with open(arch_summary_path, 'r') as f:
            results['architecture'] = json.load(f)

    # Load learning rate experiment
    lr_dir = os.path.join(base_dir, 'learning_rate')
    lr_summary_path = os.path.join(lr_dir, 'experiment_summary.json')
    if os.path.exists(lr_summary_path):
        with open(lr_summary_path, 'r') as f:
            results['learning_rate'] = json.load(f)

    return results


def create_master_comparison(results: Dict, save_dir: str):
    """
    Create comprehensive comparison visualizations across all experiments.

    Args:
        results: Dictionary containing all experiment results
        save_dir: Directory to save comparison plots
    """
    os.makedirs(save_dir, exist_ok=True)

    # =========================================================================
    # 1. GRAND SUMMARY - ALL MODELS COMPARISON
    # =========================================================================
    print("\nGenerating Grand Summary...")

    all_models = []

    # Collect baseline
    if results['baseline'].get('metrics'):
        all_models.append({
            'name': 'Baseline',
            'experiment': 'Baseline',
            'final_loss': results['baseline']['metrics']['final_total_loss'],
            'training_time': results['baseline']['metrics']['training_time'],
            'category': 'Reference'
        })

    # Collect loss weights models
    if results['loss_weights'].get('models'):
        for name, data in results['loss_weights']['models'].items():
            all_models.append({
                'name': name,
                'experiment': 'Loss Weights',
                'final_loss': data['final_total_loss'],
                'training_time': data['training_time'],
                'category': 'Loss Weights'
            })

    # Collect architecture models
    if results['architecture'].get('models'):
        for name, data in results['architecture']['models'].items():
            all_models.append({
                'name': name,
                'experiment': 'Architecture',
                'final_loss': data['final_total_loss'],
                'training_time': data['training_time'],
                'category': 'Architecture'
            })

    # Collect learning rate models
    if results['learning_rate'].get('models'):
        for name, data in results['learning_rate']['models'].items():
            all_models.append({
                'name': name,
                'experiment': 'Learning Rate',
                'final_loss': data['final_total_loss'],
                'training_time': data['training_time'],
                'category': 'Learning Rate'
            })

    if not all_models:
        print("No experiment results found!")
        return

    # Sort by final loss
    all_models.sort(key=lambda x: x['final_loss'])

    # Color mapping by category
    category_colors = {
        'Reference': '#34495e',
        'Loss Weights': '#27ae60',
        'Architecture': '#e74c3c',
        'Learning Rate': '#3498db'
    }

    # =========================================================================
    # 1a. HORIZONTAL BAR CHART - ALL MODELS RANKED BY LOSS
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, max(8, len(all_models) * 0.5)))

    y_pos = np.arange(len(all_models))
    losses = [m['final_loss'] for m in all_models]
    colors = [category_colors[m['category']] for m in all_models]
    names = [m['name'] for m in all_models]

    bars = ax.barh(y_pos, losses, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Final Total Loss', fontsize=12)
    ax.set_title('All Models Ranked by Final Loss (Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for bar, val in zip(bars, losses):
        ax.text(val * 1.1, bar.get_y() + bar.get_height()/2,
                f'{val:.2e}', va='center', fontsize=9)

    # Add legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color, label=cat)
                       for cat, color in category_colors.items()]
    ax.legend(handles=legend_elements, loc='lower right', title='Experiment Type')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '01_all_models_ranking.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 1b. SCATTER PLOT - LOSS VS TRAINING TIME
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 8))

    for category in category_colors.keys():
        models_in_cat = [m for m in all_models if m['category'] == category]
        if models_in_cat:
            times = [m['training_time'] for m in models_in_cat]
            losses = [m['final_loss'] for m in models_in_cat]
            ax.scatter(times, losses, s=200, c=category_colors[category],
                      label=category, edgecolors='black', linewidths=1.5)

            for m in models_in_cat:
                ax.annotate(m['name'].split('_')[-1], (m['training_time'], m['final_loss']),
                           textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel('Training Time (seconds)', fontsize=12)
    ax.set_ylabel('Final Total Loss', fontsize=12)
    ax.set_title('Training Efficiency: Loss vs Time Trade-off', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(title='Experiment Type')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '02_efficiency_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 2. EXPERIMENT-WISE COMPARISON
    # =========================================================================
    print("Generating Experiment-wise Comparison...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Loss Weights Impact
    if results['loss_weights'].get('models'):
        models = results['loss_weights']['models']
        names = list(models.keys())
        losses = [models[n]['final_total_loss'] for n in names]
        colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(names)))

        bars = axes[0].bar(range(len(names)), losses, color=colors, edgecolor='black')
        axes[0].set_xticks(range(len(names)))
        axes[0].set_xticklabels([n.replace('Model_', '').replace('_', '\n') for n in names],
                                fontsize=9, rotation=0)
        axes[0].set_ylabel('Final Total Loss')
        axes[0].set_title('Loss Weights Impact', fontsize=12, fontweight='bold')
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, losses):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.1e}', ha='center', va='bottom', fontsize=8)

    # Architecture Impact
    if results['architecture'].get('models'):
        models = results['architecture']['models']
        names = list(models.keys())
        losses = [models[n]['final_total_loss'] for n in names]
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(names)))

        bars = axes[1].bar(range(len(names)), losses, color=colors, edgecolor='black')
        axes[1].set_xticks(range(len(names)))
        axes[1].set_xticklabels([n.replace('Model_', '').replace('_', '\n') for n in names],
                                fontsize=9, rotation=0)
        axes[1].set_ylabel('Final Total Loss')
        axes[1].set_title('Architecture Impact', fontsize=12, fontweight='bold')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, losses):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.1e}', ha='center', va='bottom', fontsize=8)

    # Learning Rate Impact
    if results['learning_rate'].get('models'):
        models = results['learning_rate']['models']
        names = list(models.keys())
        losses = [models[n]['final_total_loss'] for n in names]
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(names)))

        bars = axes[2].bar(range(len(names)), losses, color=colors, edgecolor='black')
        axes[2].set_xticks(range(len(names)))
        axes[2].set_xticklabels([f"LR={models[n]['learning_rate']}" for n in names],
                                fontsize=9, rotation=0)
        axes[2].set_ylabel('Final Total Loss')
        axes[2].set_title('Learning Rate Impact', fontsize=12, fontweight='bold')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, losses):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.1e}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('Impact of Each Parameter on Model Performance', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '03_parameter_impact.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 3. LOSS COMPONENT BREAKDOWN
    # =========================================================================
    print("Generating Loss Component Breakdown...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    loss_components = ['final_data_loss', 'final_bc_loss', 'final_ic_loss', 'final_pde_loss']
    component_names = ['Data Loss', 'BC Loss', 'IC Loss', 'PDE Loss']
    component_colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

    # For each experiment type
    for exp_idx, (exp_name, exp_title) in enumerate([
        ('loss_weights', 'Loss Weights Experiment'),
        ('architecture', 'Architecture Experiment'),
        ('learning_rate', 'Learning Rate Experiment')
    ]):
        if not results[exp_name].get('models'):
            continue

        ax = axes[exp_idx // 2, exp_idx % 2]
        models = results[exp_name]['models']
        model_names = list(models.keys())
        x = np.arange(len(model_names))
        width = 0.2

        for i, (comp, comp_name, color) in enumerate(zip(loss_components, component_names, component_colors)):
            values = [models[m][comp] for m in model_names]
            ax.bar(x + i*width, values, width, label=comp_name, color=color, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Model')
        ax.set_ylabel('Loss Value')
        ax.set_title(exp_title, fontsize=12, fontweight='bold')
        ax.set_xticks(x + 1.5*width)
        ax.set_xticklabels([n.split('_')[-1] for n in model_names], fontsize=9)
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    # Summary in 4th subplot
    ax = axes[1, 1]

    # Best model per experiment
    best_models = []
    if results['loss_weights'].get('best_model'):
        best_models.append(('Loss Weights', results['loss_weights']['best_model']))
    if results['architecture'].get('best_model'):
        best_models.append(('Architecture', results['architecture']['best_model']))
    if results['learning_rate'].get('best_model'):
        best_models.append(('Learning Rate', results['learning_rate']['best_model']))

    ax.axis('off')
    summary_text = "BEST MODELS PER EXPERIMENT\n" + "="*40 + "\n\n"
    for exp, model in best_models:
        summary_text += f"{exp}:\n  → {model}\n\n"

    if all_models:
        summary_text += "\nOVERALL BEST MODEL\n" + "="*40 + "\n"
        summary_text += f"  → {all_models[0]['name']}\n"
        summary_text += f"  → Final Loss: {all_models[0]['final_loss']:.2e}\n"
        summary_text += f"  → Category: {all_models[0]['category']}"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Loss Component Analysis Across Experiments', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '04_loss_components.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 4. COMPREHENSIVE SUMMARY TABLE
    # =========================================================================
    print("Generating Summary Table...")

    fig, ax = plt.subplots(figsize=(20, 12))
    ax.axis('off')

    # Prepare table data
    headers = ['Experiment', 'Model', 'Configuration', 'Train Time (s)',
               'Total Loss', 'Data Loss', 'BC Loss', 'IC Loss', 'PDE Loss']

    table_data = []

    # Baseline
    if results['baseline'].get('metrics'):
        m = results['baseline']['metrics']
        table_data.append([
            'Baseline', 'Baseline', '[2,50,50,50,1], W=[1,1,1,1], LR=0.001',
            f"{m['training_time']:.1f}", f"{m['final_total_loss']:.2e}",
            f"{m['final_data_loss']:.2e}", f"{m['final_bc_loss']:.2e}",
            f"{m['final_ic_loss']:.2e}", f"{m['final_pde_loss']:.2e}"
        ])

    # Loss weights
    if results['loss_weights'].get('models'):
        for name, data in results['loss_weights']['models'].items():
            config = f"W={data['loss_weights']}"
            table_data.append([
                'Loss Weights', name.replace('Model_', ''), config,
                f"{data['training_time']:.1f}", f"{data['final_total_loss']:.2e}",
                f"{data['final_data_loss']:.2e}", f"{data['final_bc_loss']:.2e}",
                f"{data['final_ic_loss']:.2e}", f"{data['final_pde_loss']:.2e}"
            ])

    # Architecture
    if results['architecture'].get('models'):
        for name, data in results['architecture']['models'].items():
            config = f"Arch={data['layers']}"
            table_data.append([
                'Architecture', name.replace('Model_', ''), config,
                f"{data['training_time']:.1f}", f"{data['final_total_loss']:.2e}",
                f"{data['final_data_loss']:.2e}", f"{data['final_bc_loss']:.2e}",
                f"{data['final_ic_loss']:.2e}", f"{data['final_pde_loss']:.2e}"
            ])

    # Learning rate
    if results['learning_rate'].get('models'):
        for name, data in results['learning_rate']['models'].items():
            config = f"LR={data['learning_rate']}"
            table_data.append([
                'Learning Rate', name.replace('Model_', ''), config,
                f"{data['training_time']:.1f}", f"{data['final_total_loss']:.2e}",
                f"{data['final_data_loss']:.2e}", f"{data['final_bc_loss']:.2e}",
                f"{data['final_ic_loss']:.2e}", f"{data['final_pde_loss']:.2e}"
            ])

    if table_data:
        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            loc='center',
            cellLoc='center',
            colColours=['#2c3e50']*len(headers)
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_text_props(color='white', fontweight='bold')

        # Color code rows by experiment type
        exp_colors = {
            'Baseline': '#ecf0f1',
            'Loss Weights': '#d5f5e3',
            'Architecture': '#fadbd8',
            'Learning Rate': '#d6eaf8'
        }

        for i, row in enumerate(table_data):
            for j in range(len(headers)):
                table[(i+1, j)].set_facecolor(exp_colors.get(row[0], 'white'))

    plt.title('Comprehensive Results Summary - All Experiments', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(save_dir, '05_comprehensive_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 5. KEY INSIGHTS VISUALIZATION
    # =========================================================================
    print("Generating Key Insights...")

    fig = plt.figure(figsize=(20, 12))

    # Create grid specification
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 5a. Parameter Sensitivity Analysis
    ax1 = fig.add_subplot(gs[0, 0])

    sensitivities = []
    if results['loss_weights'].get('models'):
        losses = [m['final_total_loss'] for m in results['loss_weights']['models'].values()]
        sensitivities.append(('Loss Weights', max(losses)/min(losses) if min(losses) > 0 else 1))

    if results['architecture'].get('models'):
        losses = [m['final_total_loss'] for m in results['architecture']['models'].values()]
        sensitivities.append(('Architecture', max(losses)/min(losses) if min(losses) > 0 else 1))

    if results['learning_rate'].get('models'):
        losses = [m['final_total_loss'] for m in results['learning_rate']['models'].values()]
        sensitivities.append(('Learning Rate', max(losses)/min(losses) if min(losses) > 0 else 1))

    if sensitivities:
        params = [s[0] for s in sensitivities]
        ratios = [s[1] for s in sensitivities]
        bars = ax1.bar(params, ratios, color=['#27ae60', '#e74c3c', '#3498db'], edgecolor='black')
        ax1.set_ylabel('Sensitivity Ratio\n(Max/Min Loss)')
        ax1.set_title('Parameter Sensitivity\n(Higher = More Impact)', fontweight='bold')
        ax1.set_yscale('log')

        for bar, val in zip(bars, ratios):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.1f}x', ha='center', va='bottom', fontsize=10)

    # 5b. Training Time Distribution
    ax2 = fig.add_subplot(gs[0, 1])

    times_by_exp = {}
    if results['loss_weights'].get('models'):
        times_by_exp['Loss Weights'] = [m['training_time'] for m in results['loss_weights']['models'].values()]
    if results['architecture'].get('models'):
        times_by_exp['Architecture'] = [m['training_time'] for m in results['architecture']['models'].values()]
    if results['learning_rate'].get('models'):
        times_by_exp['Learning Rate'] = [m['training_time'] for m in results['learning_rate']['models'].values()]

    if times_by_exp:
        exp_names = list(times_by_exp.keys())
        ax2.boxplot([times_by_exp[e] for e in exp_names], labels=exp_names)
        ax2.set_ylabel('Training Time (seconds)')
        ax2.set_title('Training Time Distribution\nby Experiment Type', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

    # 5c. Loss Component Importance
    ax3 = fig.add_subplot(gs[0, 2])

    avg_components = {'Data': [], 'BC': [], 'IC': [], 'PDE': []}

    for exp_name in ['loss_weights', 'architecture', 'learning_rate']:
        if results[exp_name].get('models'):
            for m in results[exp_name]['models'].values():
                avg_components['Data'].append(m['final_data_loss'])
                avg_components['BC'].append(m['final_bc_loss'])
                avg_components['IC'].append(m['final_ic_loss'])
                avg_components['PDE'].append(m['final_pde_loss'])

    if avg_components['Data']:
        comp_names = list(avg_components.keys())
        avg_values = [np.mean(avg_components[c]) for c in comp_names]
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

        bars = ax3.bar(comp_names, avg_values, color=colors, edgecolor='black')
        ax3.set_ylabel('Average Final Loss')
        ax3.set_title('Average Loss by Component\n(Across All Models)', fontweight='bold')
        ax3.set_yscale('log')

        for bar, val in zip(bars, avg_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.2e}', ha='center', va='bottom', fontsize=9)

    # 5d. Top 5 Best Models
    ax4 = fig.add_subplot(gs[1, 0:2])

    if all_models:
        top5 = all_models[:min(5, len(all_models))]
        names = [m['name'] for m in top5]
        losses = [m['final_loss'] for m in top5]
        colors = [category_colors[m['category']] for m in top5]

        bars = ax4.barh(range(len(names)), losses, color=colors, edgecolor='black')
        ax4.set_yticks(range(len(names)))
        ax4.set_yticklabels(names)
        ax4.set_xlabel('Final Total Loss')
        ax4.set_title('Top 5 Best Performing Models', fontweight='bold')
        ax4.set_xscale('log')
        ax4.invert_yaxis()

        for i, (bar, val) in enumerate(zip(bars, losses)):
            ax4.text(val * 1.1, bar.get_y() + bar.get_height()/2,
                    f'{val:.2e}', va='center', fontsize=10)

        # Add medal icons
        medals = ['🥇', '🥈', '🥉', '4th', '5th']
        for i, medal in enumerate(medals[:len(names)]):
            ax4.text(-0.02, i, medal, transform=ax4.get_yaxis_transform(),
                    va='center', ha='right', fontsize=14)

    # 5e. Key Findings Text
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')

    findings = """
    KEY FINDINGS
    ════════════════════════════════

    1. LOSS WEIGHTS
       • Physics-heavy weights often improve
         PDE residual but may hurt IC/BC
       • Equal weights provide balanced training

    2. ARCHITECTURE
       • Deeper networks don't always win
       • Width vs depth trade-off exists
       • Moderate architectures often optimal

    3. LEARNING RATE
       • Too high: unstable training
       • Too low: slow convergence
       • Medium (0.001) often best compromise

    RECOMMENDATION
    ════════════════════════════════
    Start with baseline configuration,
    then fine-tune based on specific
    problem requirements.
    """

    ax5.text(0.05, 0.95, findings, transform=ax5.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('PINN Heat Equation - Key Insights & Analysis', fontsize=18, fontweight='bold')
    plt.savefig(os.path.join(save_dir, '06_key_insights.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 6. SAVE COMPREHENSIVE REPORT
    # =========================================================================
    print("Generating Report...")

    report = {
        'title': 'PINN Heat Equation Parameter Study Report',
        'experiments': {
            'baseline': 'Reference model with standard configuration',
            'loss_weights': '4 configurations testing weight impact',
            'architecture': '4 configurations testing network depth/width',
            'learning_rate': '3 configurations testing learning rate'
        },
        'all_models_ranked': [
            {'rank': i+1, 'name': m['name'], 'category': m['category'],
             'final_loss': m['final_loss'], 'training_time': m['training_time']}
            for i, m in enumerate(all_models)
        ],
        'best_per_experiment': {
            'loss_weights': results['loss_weights'].get('best_model', 'N/A'),
            'architecture': results['architecture'].get('best_model', 'N/A'),
            'learning_rate': results['learning_rate'].get('best_model', 'N/A')
        },
        'overall_best': all_models[0] if all_models else None,
        'parameter_sensitivities': dict(sensitivities) if sensitivities else {}
    }

    with open(os.path.join(save_dir, 'final_report.json'), 'w') as f:
        json.dump(report, f, indent=4)

    print(f"\nAll comparison plots saved to: {save_dir}")


def main():
    """Main function to run all comparisons."""

    # Base directory for results
    base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'results'
    )

    # Comparison output directory
    comparison_dir = os.path.join(base_dir, 'comparison')

    print("\n" + "="*70)
    print("PINN HEAT EQUATION - FINAL COMPARISON ANALYSIS")
    print("="*70)

    # Load all results
    print("\nLoading experiment results...")
    results = load_experiment_results(base_dir)

    # Check what's available
    available = []
    if results['baseline'].get('metrics'):
        available.append('Baseline')
    if results['loss_weights'].get('models'):
        available.append(f"Loss Weights ({len(results['loss_weights']['models'])} models)")
    if results['architecture'].get('models'):
        available.append(f"Architecture ({len(results['architecture']['models'])} models)")
    if results['learning_rate'].get('models'):
        available.append(f"Learning Rate ({len(results['learning_rate']['models'])} models)")

    print(f"\nAvailable experiments: {', '.join(available) if available else 'None'}")

    if not available:
        print("\nNo experiment results found!")
        print("Please run the experiments first:")
        print("  python baseline.py")
        print("  python loss_weights_experiment.py")
        print("  python architecture_experiment.py")
        print("  python learning_rate_experiment.py")
        return

    # Create comparison visualizations
    print("\nGenerating comparison visualizations...")
    create_master_comparison(results, comparison_dir)

    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    print(f"\nGenerated files in {comparison_dir}:")
    print("  01_all_models_ranking.png    - All models ranked by loss")
    print("  02_efficiency_scatter.png    - Loss vs training time")
    print("  03_parameter_impact.png      - Impact of each parameter")
    print("  04_loss_components.png       - Loss component breakdown")
    print("  05_comprehensive_summary.png - Full summary table")
    print("  06_key_insights.png          - Key findings visualization")
    print("  final_report.json            - Complete results in JSON")
    print("="*70)


if __name__ == "__main__":
    main()
