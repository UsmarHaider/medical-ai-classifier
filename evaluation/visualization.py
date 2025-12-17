"""
Visualization Module for Model Evaluation
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import os


def plot_training_history(history: Dict,
                          save_path: str = None,
                          show: bool = True) -> plt.Figure:
    """
    Plot training history (loss and accuracy curves).

    Args:
        history: Training history dictionary
        save_path: Path to save the figure
        show: Whether to display the figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss
    axes[0].plot(history.get('loss', []), label='Training Loss', linewidth=2)
    axes[0].plot(history.get('val_loss', []), label='Validation Loss', linewidth=2)
    axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot accuracy
    axes[1].plot(history.get('accuracy', []), label='Training Accuracy', linewidth=2)
    axes[1].plot(history.get('val_accuracy', []), label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: List[str] = None,
                          normalize: bool = True,
                          save_path: str = None,
                          show: bool = True,
                          cmap: str = 'Blues') -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize the matrix
        save_path: Path to save the figure
        show: Whether to display the figure
        cmap: Colormap to use

    Returns:
        Matplotlib figure
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names or range(cm.shape[1]),
        yticklabels=class_names or range(cm.shape[0]),
        ax=ax,
        annot_kws={'size': 12}
    )

    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_roc_curve(y_true: np.ndarray,
                   y_prob: np.ndarray,
                   class_names: List[str] = None,
                   save_path: str = None,
                   show: bool = True) -> plt.Figure:
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        class_names: List of class names
        save_path: Path to save the figure
        show: Whether to display the figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    num_classes = len(np.unique(y_true))

    if num_classes == 2:
        # Binary classification
        probs = y_prob.flatten() if y_prob.shape[-1] == 1 else y_prob[:, 1]
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    else:
        # Multi-class: plot ROC for each class
        colors = plt.cm.Set1(np.linspace(0, 1, num_classes))

        for i in range(num_classes):
            binary_labels = (y_true == i).astype(int)
            if len(np.unique(binary_labels)) < 2:
                continue

            fpr, tpr, _ = roc_curve(binary_labels, y_prob[:, i])
            roc_auc = auc(fpr, tpr)

            class_name = class_names[i] if class_names and i < len(class_names) else f'Class {i}'
            ax.plot(fpr, tpr, color=colors[i], linewidth=2,
                   label=f'{class_name} (AUC = {roc_auc:.3f})')

    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_precision_recall_curve(y_true: np.ndarray,
                                 y_prob: np.ndarray,
                                 class_names: List[str] = None,
                                 save_path: str = None,
                                 show: bool = True) -> plt.Figure:
    """
    Plot precision-recall curve.

    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        class_names: List of class names
        save_path: Path to save the figure
        show: Whether to display the figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    num_classes = len(np.unique(y_true))

    if num_classes == 2:
        probs = y_prob.flatten() if y_prob.shape[-1] == 1 else y_prob[:, 1]
        precision, recall, _ = precision_recall_curve(y_true, probs)
        ax.plot(recall, precision, linewidth=2, label='Precision-Recall')
    else:
        colors = plt.cm.Set1(np.linspace(0, 1, num_classes))

        for i in range(num_classes):
            binary_labels = (y_true == i).astype(int)
            if len(np.unique(binary_labels)) < 2:
                continue

            precision, recall, _ = precision_recall_curve(binary_labels, y_prob[:, i])
            class_name = class_names[i] if class_names and i < len(class_names) else f'Class {i}'
            ax.plot(recall, precision, color=colors[i], linewidth=2, label=class_name)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_metrics_comparison(metrics_dict: Dict[str, Dict],
                            metric_names: List[str] = None,
                            save_path: str = None,
                            show: bool = True) -> plt.Figure:
    """
    Plot comparison of metrics across multiple models.

    Args:
        metrics_dict: Dictionary of model_name -> metrics
        metric_names: List of metric names to compare
        save_path: Path to save the figure
        show: Whether to display the figure

    Returns:
        Matplotlib figure
    """
    if metric_names is None:
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score']

    model_names = list(metrics_dict.keys())
    x = np.arange(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, metric in enumerate(metric_names):
        values = [metrics_dict[model].get(metric, 0) for model in model_names]
        ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(metric_names) - 1) / 2)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_class_distribution(labels: np.ndarray,
                            class_names: List[str] = None,
                            save_path: str = None,
                            show: bool = True) -> plt.Figure:
    """
    Plot class distribution.

    Args:
        labels: Array of labels
        class_names: List of class names
        save_path: Path to save the figure
        show: Whether to display the figure

    Returns:
        Matplotlib figure
    """
    # Get unique classes and counts
    unique, counts = np.unique(labels, return_counts=True)

    if class_names is None:
        class_names = [f'Class {i}' for i in unique]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(class_names, counts, color=plt.cm.Set2(np.linspace(0, 1, len(unique))))

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
               str(count), ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def create_evaluation_report(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              y_prob: np.ndarray,
                              class_names: List[str],
                              history: Dict = None,
                              output_dir: str = 'evaluation_report') -> str:
    """
    Create a complete evaluation report with all visualizations.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        class_names: List of class names
        history: Training history (optional)
        output_dir: Directory to save report

    Returns:
        Path to output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot training history if available
    if history:
        plot_training_history(
            history,
            save_path=os.path.join(output_dir, 'training_history.png'),
            show=False
        )

    # Plot confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=os.path.join(output_dir, 'confusion_matrix.png'),
        show=False
    )

    # Plot ROC curve
    plot_roc_curve(
        y_true, y_prob, class_names,
        save_path=os.path.join(output_dir, 'roc_curve.png'),
        show=False
    )

    # Plot precision-recall curve
    plot_precision_recall_curve(
        y_true, y_prob, class_names,
        save_path=os.path.join(output_dir, 'precision_recall_curve.png'),
        show=False
    )

    # Plot class distribution
    plot_class_distribution(
        y_true, class_names,
        save_path=os.path.join(output_dir, 'class_distribution.png'),
        show=False
    )

    print(f"Evaluation report saved to: {output_dir}")
    return output_dir
