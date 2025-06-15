import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_model_results(json_files):
    """Helper function to load model results from JSON files."""
    results = []
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return results

def plot_performance_vs_model_size(json_files, save_path=None):
    """
    Plot validation accuracy vs model parameters.
    
    Objective: Show the relationship between model size and performance,
    demonstrating that smaller models achieve better accuracy on limited data.
    """
    results = load_model_results(json_files)
    
    # Extract data
    model_versions = [r['model_version'] for r in results]
    params = [r['params'] / 1e6 for r in results]  # Convert to millions
    val_accuracies = [r['best_val_acc'] * 100 for r in results]  # Convert to percentage
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(params, val_accuracies, 'bo-', linewidth=2, markersize=8)
    
    # Annotate points
    for i, version in enumerate(model_versions):
        plt.annotate(f'B{version}', (params[i], val_accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    plt.xlabel('Model Parameters (Millions)')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.title('Performance vs Model Size\n(Showing Optimal Complexity for Limited Data)')
    plt.grid(True, alpha=0.3)
    
    # Highlight the peak
    max_idx = np.argmax(val_accuracies)
    plt.scatter(params[max_idx], val_accuracies[max_idx], 
               color='red', s=100, alpha=0.7, label=f'Peak: B{model_versions[max_idx]}')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_parameter_efficiency(json_files, save_path=None):
    """
    Plot accuracy per million parameters for each model.
    
    Objective: Demonstrate parameter efficiency, showing that smaller models
    achieve more accuracy per parameter than larger models.
    """
    results = load_model_results(json_files)
    
    # Extract data
    model_versions = [f"B{r['model_version']}" for r in results]
    params = [r['params'] / 1e6 for r in results]
    val_accuracies = [r['best_val_acc'] * 100 for r in results]
    efficiency = [acc / param for acc, param in zip(val_accuracies, params)]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_versions, efficiency, 
                   color=['green' if e > 10 else 'orange' if e > 5 else 'red' for e in efficiency])
    
    # Add value labels on bars
    for bar, eff in zip(bars, efficiency):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{eff:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Model Version')
    plt.ylabel('Accuracy Points per Million Parameters')
    plt.title('Parameter Efficiency Comparison\n(Higher is Better)')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add efficiency threshold line
    plt.axhline(y=10, color='green', linestyle='--', alpha=0.7, label='High Efficiency (>10)')
    plt.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Medium Efficiency (5-10)')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_stability(json_files, save_path=None):
    """
    Plot training stability metrics with NaN failure indicators.
    
    Objective: Show that larger models are harder to train and less stable,
    with clear indicators for training failures (NaN values).
    """
    results = load_model_results(json_files)
    
    # Extract data
    model_versions = [f"B{r['model_version']}" for r in results]
    epochs_completed = [r['epochs_trained'] for r in results]
    
    # Handle NaN values in validation loss
    final_val_loss = []
    has_nan = []
    for r in results:
        val_loss = r['training_history']['val_loss']

        final_val_loss_value = val_loss[-1]
        if np.isnan(final_val_loss_value):
            # Use the last valid loss or mark as failure
            val_loss_history = r['training_history']['val_loss']
            last_valid_loss = next((loss for loss in reversed(val_loss) if not np.isnan(loss)), 5.0)
            final_val_loss.append(last_valid_loss)
            has_nan.append(True)
        else:
            final_val_loss.append(final_val_loss_value)
            has_nan.append(False)
    
    training_times = [r['training_time_seconds'] / 3600 for r in results]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Epochs completed with failure indicators
    colors = ['red' if nan else 'skyblue' for nan in has_nan]
    bars1 = ax1.bar(model_versions, epochs_completed, color=colors)
    ax1.set_ylabel('Epochs Completed')
    ax1.set_title('Training Duration (Red = Training Failure)')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, epochs, nan_flag in zip(bars1, epochs_completed, has_nan):
        label = f'{epochs}*' if nan_flag else f'{epochs}'
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                label, ha='center', va='bottom', 
                fontweight='bold' if nan_flag else 'normal')
    
    # Plot 2: Final validation loss with failure indicators
    colors = ['red' if nan else 'lightcoral' for nan in has_nan]
    bars2 = ax2.bar(model_versions, final_val_loss, color=colors)
    ax2.set_ylabel('Final Validation Loss')
    ax2.set_title('Training Stability (Red = NaN Failure)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add failure annotations
    for bar, loss, nan_flag in zip(bars2, final_val_loss, has_nan):
        if nan_flag:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    'NaN', ha='center', va='bottom', 
                    fontweight='bold', color='white')
    
    # Plot 3: Training time
    bars3 = ax3.bar(model_versions, training_times, color='lightgreen')
    ax3.set_ylabel('Training Time (Hours)')
    ax3.set_title('Computational Efficiency')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Training success indicator
    success_scores = [0 if nan else 1 for nan in has_nan]
    colors = ['red' if score == 0 else 'green' for score in success_scores]
    bars4 = ax4.bar(model_versions, success_scores, color=colors)
    ax4.set_ylabel('Training Success (1=Success, 0=Failure)')
    ax4.set_title('Training Stability Summary')
    ax4.set_ylim(-0.1, 1.1)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add success/failure labels
    for bar, score in zip(bars4, success_scores):
        label = 'SUCCESS' if score == 1 else 'FAILED'
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                label, ha='center', va='bottom', fontweight='bold', color='white')
    
    plt.suptitle('Training Stability Analysis (Red = NaN Failures)', fontsize=16, y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_learning_curves(json_files, save_path=None):
    """
    Plot validation accuracy curves with NaN failure indicators.
    """
    results = load_model_results(json_files)
    
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    for i, result in enumerate(results):
        model_version = f"B{result['model_version']}"
        val_accuracies = [acc * 100 for acc in result['training_history']['val_accuracy']]
        val_losses = result['training_history']['val_loss']
        epochs = list(range(1, len(val_accuracies) + 1))
        params = result['params'] / 1e6
        
        # Check for NaN failures
        nan_epochs = [i+1 for i, loss in enumerate(val_losses) if np.isnan(loss)]
        has_failure = len(nan_epochs) > 0
        
        # Plot main curve
        line_style = '--' if has_failure else '-'
        plt.plot(epochs, val_accuracies, 
                color=colors[i % len(colors)], 
                linewidth=2, 
                linestyle=line_style,
                marker='o', 
                markersize=4,
                label=f'{model_version} ({params:.1f}M params){"*" if has_failure else ""}')
        
        # Mark NaN failure points
        if nan_epochs:
            nan_accuracies = [val_accuracies[epoch-1] for epoch in nan_epochs if epoch-1 < len(val_accuracies)]
            if nan_accuracies:
                plt.scatter(nan_epochs[:len(nan_accuracies)], nan_accuracies, 
                           color='red', s=100, marker='X', 
                           edgecolor='black', linewidth=2, 
                           label=f'{model_version} NaN Failures' if i == 0 else "")
        
        # Mark best performance (before any NaN)
        valid_epochs = [i for i, loss in enumerate(val_losses) if not np.isnan(loss)]
        if valid_epochs:
            valid_accuracies = [val_accuracies[i] for i in valid_epochs]
            best_idx = np.argmax(valid_accuracies)
            best_epoch = valid_epochs[best_idx] + 1
            best_acc = valid_accuracies[best_idx]
            
            plt.scatter(best_epoch, best_acc, 
                       color=colors[i % len(colors)], 
                       s=100, 
                       marker='*', 
                       edgecolor='black', 
                       linewidth=1)
    
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Learning Curves Comparison\n(* = NaN failures, X = failure points, Stars = peak performance)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Add failure indicator in legend
    plt.scatter([], [], color='red', s=100, marker='X', 
               edgecolor='black', linewidth=2, label='NaN Failure Points')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_cost_benefit_analysis(json_files, save_path=None):
    """
    Plot accuracy vs training time with parameter count as bubble size.
    
    Objective: Show the cost-benefit trade-off, highlighting that smaller models
    achieve better accuracy in less time with fewer parameters.
    """
    results = load_model_results(json_files)
    
    # Extract data
    model_versions = [f"B{r['model_version']}" for r in results]
    training_times = [r['training_time_seconds'] / 3600 for r in results]  # Hours
    val_accuracies = [r['best_val_acc'] * 100 for r in results]
    params = [r['params'] / 1e6 for r in results]  # Millions
    
    plt.figure(figsize=(10, 8))
    
    # Create bubble plot
    colors = ['green', 'blue', 'orange', 'red', 'purple', 'cyan', 'magenta']
    for i, (version, time, acc, param) in enumerate(zip(model_versions, training_times, val_accuracies, params)):
        plt.scatter(time, acc, 
                   s=param * 50,  # Bubble size proportional to parameters
                   alpha=0.6, 
                   color=colors[i % len(colors)],
                   label=f'{version} ({param:.1f}M params)')
        
        # Annotate with model version
        plt.annotate(version, (time, acc), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontweight='bold', fontsize=10)
    
    plt.xlabel('Training Time (Hours)')
    plt.ylabel('Best Validation Accuracy (%)')
    plt.title('Cost-Benefit Analysis\n(Bubble size = Parameter count)')
    
    # Add quadrant lines
    mean_time = np.mean(training_times)
    mean_acc = np.mean(val_accuracies)
    plt.axvline(x=mean_time, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=mean_acc, color='gray', linestyle='--', alpha=0.5)
    
    # Label quadrants - corrected positioning
    time_range = max(training_times) - min(training_times)
    acc_range = max(val_accuracies) - min(val_accuracies)
    
    # Top-left quadrant (High Accuracy, Low Time - OPTIMAL)
    plt.text(min(training_times) + time_range * 0.15, 
            max(val_accuracies) - acc_range * 0.05, 
            'High Accuracy\nLow Time\n(OPTIMAL)', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
            fontsize=9, ha='center', va='top')
    
    # Optional: Add other quadrant labels for clarity
    # Top-right quadrant (High Accuracy, High Time)
    plt.text(max(training_times) - time_range * 0.15, 
            max(val_accuracies) - acc_range * 0.1,
            'High Accuracy\nHigh Time', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            fontsize=9, ha='center', va='top')
    
    # Bottom-left quadrant (Low Accuracy, Low Time)
    plt.text(min(training_times) + time_range * 0.15, 
            min(val_accuracies) + acc_range * 0.15,
            'Low Accuracy\nLow Time', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7),
            fontsize=9, ha='center', va='bottom')
    
    # Bottom-right quadrant (Low Accuracy, High Time - WORST)
    plt.text(max(training_times) - time_range * 0.15, 
            min(val_accuracies) + acc_range * 0.15,
            'Low Accuracy\nHigh Time\n(WORST)', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
            fontsize=9, ha='center', va='bottom')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_generalization_gap(json_files, save_path=None):
    """
    Plot generalization gap with NaN failure indicators.
    """
    results = load_model_results(json_files)
    
    # Extract data and handle NaN
    model_versions = [f"B{r['model_version']}" for r in results]
    train_acc = [r['final_metrics']['train_acc'] * 100 for r in results]
    val_acc = [r['final_metrics']['val_acc'] * 100 for r in results]
    
    # Check for NaN in validation loss (indicator of training failure)
    has_nan = []
    for r in results:
        val_loss_history = r['training_history']['val_loss']
        # Check if any value in the history is NaN
        has_nan_in_history = any(np.isnan(loss) for loss in val_loss_history)
        has_nan.append(has_nan_in_history)
    
    generalization_gap = [train - val for train, val in zip(train_acc, val_acc)]
    
    # Create subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Training vs Validation Accuracy with failure indicators
    x = np.arange(len(model_versions))
    width = 0.35
    
    # Color code for failures
    train_colors = ['red' if nan else 'skyblue' for nan in has_nan]
    val_colors = ['darkred' if nan else 'lightcoral' for nan in has_nan]
    
    bars1 = ax1.bar(x - width/2, train_acc, width, label='Training Accuracy', color=train_colors)
    bars2 = ax1.bar(x + width/2, val_acc, width, label='Validation Accuracy', color=val_colors)
    
    ax1.set_xlabel('Model Version')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Training vs Validation Accuracy\n(Red = NaN Failures)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{mv}{'*' if nan else ''}" for mv, nan in zip(model_versions, has_nan)])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Generalization Gap with failure indicators
    colors = ['darkred' if nan else 'green' if gap < 2 else 'orange' if gap < 5 else 'red' 
             for gap, nan in zip(generalization_gap, has_nan)]
    bars3 = ax2.bar(model_versions, generalization_gap, color=colors)
    
    ax2.set_xlabel('Model Version')
    ax2.set_ylabel('Generalization Gap (%)')
    ax2.set_title('Generalization Gap (Train - Val)\n(Dark Red = NaN Failures)')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels with failure indicators
    for bar, gap, nan_flag in zip(bars3, generalization_gap, has_nan):
        label = f'{gap:.1f}%*' if nan_flag else f'{gap:.1f}%'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                label, ha='center', va='bottom', 
                fontweight='bold' if nan_flag else 'normal')
    
    # Add failure legend
    ax1.bar([], [], color='red', label='Training Failure (NaN)')
    ax1.legend()
    
    plt.suptitle('Generalization Analysis (* = Training Failures)', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_models_training_history(json_files, save_path=None):
    """
    Plot training and validation accuracy/loss curves for individual models.
    
    Args:
        json_files (list): List of JSON file paths containing model results
        save_path (str): Optional path template for saving plots (will append model version)
    
    Objective: Show detailed training dynamics including accuracy and loss
    progression over epochs for each model individually.
    """
    results = load_model_results(json_files)
    
    if not results:
        print("No valid results found")
        return
    
    for result in results:
        # Extract training history
        history = result['training_history']
        model_version = result['model_version']
        params = result['params'] / 1e6  # Convert to millions
        
        train_acc = [acc * 100 for acc in history['accuracy']]  # Convert to percentage
        val_acc = [acc * 100 for acc in history['val_accuracy']]
        train_loss = history['loss']
        val_loss = history['val_loss']
        
        epochs = list(range(1, len(train_acc) + 1))
        
        # Create subplots for this model
        plt.figure(figsize=(14, 5))
        
        # Subplot 1: Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_acc, 'b-', linewidth=2, label='Training')
        plt.plot(epochs, val_acc, 'r-', linewidth=2, label='Validation')
        plt.title(f'B{model_version} Accuracy ({params:.1f}M params)')
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        # Mark best validation accuracy
        best_val_epoch = np.argmax(val_acc) + 1
        best_val_acc = max(val_acc)
        plt.scatter(best_val_epoch, best_val_acc, color='red', s=120, marker='*', 
                   edgecolor='black', linewidth=1, zorder=5)
        plt.annotate(f'Peak: {best_val_acc:.1f}%\nEpoch {best_val_epoch}', 
                    xy=(best_val_epoch, best_val_acc), 
                    xytext=(10, -20), textcoords='offset points', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    fontsize=9, ha='left')
        
        # Add final accuracy info
        final_train_acc = train_acc[-1]
        final_val_acc = val_acc[-1]
        plt.text(0.02, 0.98, f'Final Train: {final_train_acc:.1f}%\nFinal Val: {final_val_acc:.1f}%', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                fontsize=9)
        
        # Subplot 2: Loss
        plt.subplot(1, 2, 2)
        
        # Handle NaN values in loss
        train_loss_clean = []
        val_loss_clean = []
        valid_epochs = []
        nan_epochs = []
        
        for i, (t_loss, v_loss) in enumerate(zip(train_loss, val_loss)):
            if np.isnan(v_loss) or np.isnan(t_loss):
                nan_epochs.append(i + 1)
            else:
                train_loss_clean.append(t_loss)
                val_loss_clean.append(v_loss)
                valid_epochs.append(i + 1)
        
        # Plot loss curves
        if valid_epochs:
            plt.plot(valid_epochs, train_loss_clean, 'b-', linewidth=2, label='Training')
            plt.plot(valid_epochs, val_loss_clean, 'r-', linewidth=2, label='Validation')
        
        # Mark NaN points if any
        if nan_epochs:
            nan_y_pos = [max(train_loss_clean + val_loss_clean) * 1.1] * len(nan_epochs)
            plt.scatter(nan_epochs, nan_y_pos, color='red', s=120, 
                       marker='X', edgecolor='black', linewidth=2, 
                       label='NaN Failures', zorder=5)
            plt.text(0.02, 0.98, f'NaN Failures: {len(nan_epochs)} epochs', 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
                    fontsize=9)
        
        plt.title(f'B{model_version} Loss ({params:.1f}M params)')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        # Mark best validation loss (minimum)
        if val_loss_clean:
            best_loss = min(val_loss_clean)
            best_loss_epoch = valid_epochs[val_loss_clean.index(best_loss)]
            plt.scatter(best_loss_epoch, best_loss, color='red', s=120, marker='*', 
                       edgecolor='black', linewidth=1, zorder=5)
            plt.annotate(f'Best: {best_loss:.3f}\nEpoch {best_loss_epoch}', 
                        xy=(best_loss_epoch, best_loss), 
                        xytext=(10, 20), textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                        fontsize=9, ha='left')
        
        # Add training summary
        epochs_completed = len(epochs)
        training_time = result.get('training_time_seconds', 0) / 3600  # Convert to hours
        
        plt.suptitle(f'EfficientNet B{model_version} Training History\n'
                    f'Completed: {epochs_completed} epochs | '
                    f'Training Time: {training_time:.1f}h | '
                    f'Parameters: {params:.1f}M', 
                    fontsize=14, y=1.05)
        
        plt.tight_layout()
        
        # Save individual plots if save_path provided
        if save_path:
            # Create filename with model version
            file_path = save_path.replace('.png', f'_B{model_version}.png')
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
        # Print summary statistics
        print(f"\nB{model_version} Training Summary:")
        print(f"  Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_val_epoch})")
        print(f"  Final Validation Accuracy: {final_val_acc:.2f}%")
        if val_loss_clean:
            print(f"  Best Validation Loss: {min(val_loss_clean):.4f}")
        if nan_epochs:
            print(f"  NaN Failures: {len(nan_epochs)} epochs - {nan_epochs}")
        print(f"  Total Training Time: {training_time:.1f} hours")
        print("-" * 50)

def plot_combined_training_history(json_files, save_path=None):
    """
    Plot combined training history for multiple models.
    
    Objective: Compare training dynamics across different model variants
    showing both accuracy and loss progression.
    """
    results = load_model_results(json_files)
    
    if not results:
        print("No valid results found")
        return
    
    plt.figure(figsize=(15, 6))
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink']
    
    # Subplot 1: Accuracy comparison
    plt.subplot(1, 2, 1)
    for i, result in enumerate(results):
        model_version = f"B{result['model_version']}"
        history = result['training_history']
        
        train_acc = [acc * 100 for acc in history['accuracy']]
        val_acc = [acc * 100 for acc in history['val_accuracy']]
        epochs = list(range(1, len(train_acc) + 1))
        
        color = colors[i % len(colors)]
        plt.plot(epochs, train_acc, color=color, linestyle='-', alpha=0.7, 
                linewidth=1.5, label=f'{model_version} Train')
        plt.plot(epochs, val_acc, color=color, linestyle='--', alpha=0.9, 
                linewidth=2, label=f'{model_version} Val')
        
        # Mark best validation accuracy
        best_val_acc = max(val_acc)
        best_epoch = val_acc.index(best_val_acc) + 1
        plt.scatter(best_epoch, best_val_acc, color=color, s=60, marker='*', 
                   edgecolor='black', linewidth=0.5, zorder=5)
    
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Loss comparison
    plt.subplot(1, 2, 2)
    for i, result in enumerate(results):
        model_version = f"B{result['model_version']}"
        history = result['training_history']
        
        train_loss = history['loss']
        val_loss = history['val_loss']
        epochs = list(range(1, len(train_loss) + 1))
        
        # Handle NaN values
        train_loss_clean = [loss if not np.isnan(loss) else None for loss in train_loss]
        val_loss_clean = [loss if not np.isnan(loss) else None for loss in val_loss]
        
        color = colors[i % len(colors)]
        plt.plot(epochs, train_loss_clean, color=color, linestyle='-', alpha=0.7, 
                linewidth=1.5, label=f'{model_version} Train')
        plt.plot(epochs, val_loss_clean, color=color, linestyle='--', alpha=0.9, 
                linewidth=2, label=f'{model_version} Val')
        
        # Mark NaN failures
        nan_epochs = [j+1 for j, loss in enumerate(val_loss) if np.isnan(loss)]
        if nan_epochs:
            plt.scatter(nan_epochs, [5.0] * len(nan_epochs), color=color, s=60, 
                       marker='X', edgecolor='black', linewidth=1, alpha=0.8)
    
    plt.title('Model Loss Comparison')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Training History Comparison Across Models', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_transfer_learning_history(json_files, save_path=None):
    """
    Plot training history specifically for transfer learning scenarios
    (combining initial training + fine-tuning phases).
    
    Objective: Show the two-phase training pattern typical in transfer learning.
    """
    results = load_model_results(json_files)
    
    plt.figure(figsize=(12, 4))
    
    for result in results:
        model_version = f"B{result['model_version']}"
        history = result['training_history']
        
        train_acc = [acc * 100 for acc in history['accuracy']]
        val_acc = [acc * 100 for acc in history['val_accuracy']]
        train_loss = history['loss']
        val_loss = history['val_loss']
        
        epochs = list(range(1, len(train_acc) + 1))
        
        # Subplot 1: Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_acc, 'b-', linewidth=2, label='Train')
        plt.plot(epochs, val_acc, 'r-', linewidth=2, label='Validation')
        
        # If there's a fine-tuning phase (detected by sudden jumps), mark it
        acc_diffs = np.diff(val_acc)
        fine_tune_start = None
        for i, diff in enumerate(acc_diffs):
            if diff > 10:  # Significant jump indicating fine-tuning start
                fine_tune_start = i + 2
                break
        
        if fine_tune_start:
            plt.axvline(x=fine_tune_start, color='green', linestyle=':', alpha=0.7, 
                       label=f'Fine-tuning starts (epoch {fine_tune_start})')
        
        plt.title(f'{model_version} - Accuracy')
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Loss
        plt.subplot(1, 2, 2)
        val_loss_clean = [loss if not np.isnan(loss) else None for loss in val_loss]
        plt.plot(epochs, train_loss, 'b-', linewidth=2, label='Train')
        plt.plot(epochs, val_loss_clean, 'r-', linewidth=2, label='Validation')
        
        if fine_tune_start:
            plt.axvline(x=fine_tune_start, color='green', linestyle=':', alpha=0.7, 
                       label=f'Fine-tuning starts (epoch {fine_tune_start})')
        
        plt.title(f'{model_version} - Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Example usage
if __name__ == "__main__":
    # List of JSON files
    json_files = [
        # "/home/veysel/dev-projects/StayAwake-AI/models_inattention/B0/model_results_B0.json",
        # "/home/veysel/dev-projects/StayAwake-AI/models_inattention/B0_without_normalizatized_images/model_results_B0.json",
        "/home/veysel/dev-projects/StayAwake-AI/models_inattention/B0_without_normalized_images_16_batches/model_results_B0.json"

        # "/home/veysel/dev-projects/StayAwake-AI/models_inattention/B1/model_results_B1.json", 
        # "/home/veysel/dev-projects/StayAwake-AI/models_inattention/B2/model_results_B2.json",
        # "/home/veysel/dev-projects/StayAwake-AI/models_inattention/B3/model_results_B3.json",
        # "/home/veysel/dev-projects/StayAwake-AI/models_inattention/B4/model_results_B4.json",

        # "/home/veysel/dev-projects/StayAwake-AI/models_inattention/Residual_B4/model_results_B4.json",
        # "/home/veysel/dev-projects/StayAwake-AI/models_inattention/ImageNet_Pretrained_B4/model_results_B4.json",
    ]
    
    # Generate all plots
    save_path='/home/veysel/dev-projects/StayAwake-AI/diagrams/{}'
    # plot_performance_vs_model_size(json_files, save_path=save_path.format('performance_vs_model_size.png'))
    # plot_parameter_efficiency(json_files, save_path=save_path.format('parameter_efficiency.png'))
    # plot_training_stability(json_files, save_path=save_path.format('training_stability.png'))
    plot_learning_curves(json_files, save_path=save_path.format('learning_curves.png'))
    # plot_cost_benefit_analysis(json_files, save_path=save_path.format('cost_benefit_analysis.png'))
    # plot_generalization_gap(json_files, save_path=save_path.format('generalization_gap.png'))

    plot_models_training_history(json_files, save_path=save_path.format('individual_training_history.png'))
    # plot_combined_training_history(json_files, save_path=save_path.format('combined_training_history.png'))
    # plot_transfer_learning_history(json_files, save_path=save_path.format('transfer_learning_history.png'))