"""
Evaluation Metrics Module for Medical Image Classification
"""
import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)
import tensorflow as tf
from tensorflow import keras


class Evaluator:
    """
    Comprehensive evaluation class for medical image classification models.
    """

    def __init__(self, model: keras.Model, class_names: List[str] = None):
        """
        Initialize the evaluator.

        Args:
            model: Trained Keras model
            class_names: List of class names
        """
        self.model = model
        self.class_names = class_names or []
        self.predictions = None
        self.probabilities = None
        self.true_labels = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input data."""
        self.probabilities = self.model.predict(X)

        # Convert probabilities to class predictions
        if self.probabilities.shape[-1] == 1:
            # Binary classification
            self.predictions = (self.probabilities > 0.5).astype(int).flatten()
        else:
            # Multi-class classification
            self.predictions = np.argmax(self.probabilities, axis=1)

        return self.predictions

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Comprehensive model evaluation.

        Args:
            X: Input data
            y: True labels

        Returns:
            Dictionary of evaluation metrics
        """
        # Store true labels
        if len(y.shape) > 1 and y.shape[-1] > 1:
            self.true_labels = np.argmax(y, axis=1)
        else:
            self.true_labels = y.flatten()

        # Make predictions
        self.predict(X)

        # Compute metrics
        metrics = self._compute_all_metrics()

        return metrics

    def _compute_all_metrics(self) -> Dict:
        """Compute all evaluation metrics."""
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(self.true_labels, self.predictions)

        # Determine if binary or multi-class
        num_classes = len(np.unique(self.true_labels))
        average = 'binary' if num_classes == 2 else 'weighted'

        # Precision, Recall, F1
        metrics['precision'] = precision_score(
            self.true_labels, self.predictions, average=average, zero_division=0
        )
        metrics['recall'] = recall_score(
            self.true_labels, self.predictions, average=average, zero_division=0
        )
        metrics['f1_score'] = f1_score(
            self.true_labels, self.predictions, average=average, zero_division=0
        )

        # Per-class metrics
        metrics['precision_per_class'] = precision_score(
            self.true_labels, self.predictions, average=None, zero_division=0
        ).tolist()
        metrics['recall_per_class'] = recall_score(
            self.true_labels, self.predictions, average=None, zero_division=0
        ).tolist()
        metrics['f1_per_class'] = f1_score(
            self.true_labels, self.predictions, average=None, zero_division=0
        ).tolist()

        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(
            self.true_labels, self.predictions
        ).tolist()

        # AUC-ROC (if probabilities available)
        if self.probabilities is not None:
            try:
                if num_classes == 2:
                    probs = self.probabilities.flatten() if self.probabilities.shape[-1] == 1 else self.probabilities[:, 1]
                    metrics['auc_roc'] = roc_auc_score(self.true_labels, probs)
                else:
                    # Multi-class AUC
                    metrics['auc_roc'] = roc_auc_score(
                        self.true_labels, self.probabilities,
                        multi_class='ovr', average='weighted'
                    )
            except Exception as e:
                metrics['auc_roc'] = None
                metrics['auc_roc_error'] = str(e)

        # Specificity (for binary classification)
        if num_classes == 2:
            cm = confusion_matrix(self.true_labels, self.predictions)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
                metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0

        return metrics

    def get_classification_report(self) -> str:
        """Get detailed classification report."""
        if self.predictions is None or self.true_labels is None:
            raise ValueError("Run evaluate() first")

        target_names = self.class_names if self.class_names else None

        return classification_report(
            self.true_labels,
            self.predictions,
            target_names=target_names,
            zero_division=0
        )

    def get_roc_data(self) -> Dict:
        """Get ROC curve data for plotting."""
        if self.probabilities is None or self.true_labels is None:
            raise ValueError("Run evaluate() first")

        num_classes = len(np.unique(self.true_labels))

        if num_classes == 2:
            probs = self.probabilities.flatten() if self.probabilities.shape[-1] == 1 else self.probabilities[:, 1]
            fpr, tpr, thresholds = roc_curve(self.true_labels, probs)
            return {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            }
        else:
            # Multi-class: compute ROC for each class
            roc_data = {}
            for i in range(num_classes):
                binary_labels = (self.true_labels == i).astype(int)
                if len(np.unique(binary_labels)) < 2:
                    continue
                fpr, tpr, thresholds = roc_curve(binary_labels, self.probabilities[:, i])
                class_name = self.class_names[i] if i < len(self.class_names) else f"Class {i}"
                roc_data[class_name] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist()
                }
            return roc_data

    def get_precision_recall_data(self) -> Dict:
        """Get precision-recall curve data for plotting."""
        if self.probabilities is None or self.true_labels is None:
            raise ValueError("Run evaluate() first")

        num_classes = len(np.unique(self.true_labels))

        if num_classes == 2:
            probs = self.probabilities.flatten() if self.probabilities.shape[-1] == 1 else self.probabilities[:, 1]
            precision, recall, thresholds = precision_recall_curve(self.true_labels, probs)
            ap = average_precision_score(self.true_labels, probs)
            return {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'average_precision': ap
            }
        else:
            pr_data = {}
            for i in range(num_classes):
                binary_labels = (self.true_labels == i).astype(int)
                if len(np.unique(binary_labels)) < 2:
                    continue
                precision, recall, _ = precision_recall_curve(binary_labels, self.probabilities[:, i])
                ap = average_precision_score(binary_labels, self.probabilities[:, i])
                class_name = self.class_names[i] if i < len(self.class_names) else f"Class {i}"
                pr_data[class_name] = {
                    'precision': precision.tolist(),
                    'recall': recall.tolist(),
                    'average_precision': ap
                }
            return pr_data


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: np.ndarray = None) -> Dict:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Flatten if needed
    if len(y_true.shape) > 1 and y_true.shape[-1] > 1:
        y_true = np.argmax(y_true, axis=1)
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Determine if binary or multi-class
    num_classes = len(np.unique(y_true))
    average = 'binary' if num_classes == 2 else 'weighted'

    # Basic metrics
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['precision'] = float(precision_score(y_true, y_pred, average=average, zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
    metrics['f1_score'] = float(f1_score(y_true, y_pred, average=average, zero_division=0))

    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

    # AUC-ROC if probabilities provided
    if y_prob is not None:
        try:
            if num_classes == 2:
                probs = y_prob.flatten() if y_prob.shape[-1] == 1 else y_prob[:, 1]
                metrics['auc_roc'] = float(roc_auc_score(y_true, probs))
            else:
                metrics['auc_roc'] = float(roc_auc_score(
                    y_true, y_prob, multi_class='ovr', average='weighted'
                ))
        except Exception:
            metrics['auc_roc'] = None

    return metrics


def evaluate_model(model: keras.Model,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   class_names: List[str] = None) -> Dict:
    """
    Evaluate a model and return comprehensive metrics.

    Args:
        model: Trained Keras model
        X_test: Test input data
        y_test: Test labels
        class_names: List of class names

    Returns:
        Dictionary of evaluation metrics
    """
    evaluator = Evaluator(model, class_names)
    metrics = evaluator.evaluate(X_test, y_test)

    # Add classification report
    metrics['classification_report'] = evaluator.get_classification_report()

    return metrics


def compare_models(models: Dict[str, keras.Model],
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   class_names: List[str] = None) -> Dict:
    """
    Compare multiple models on the same test data.

    Args:
        models: Dictionary of model name -> model
        X_test: Test input data
        y_test: Test labels
        class_names: List of class names

    Returns:
        Dictionary of comparison results
    """
    results = {}

    for name, model in models.items():
        print(f"Evaluating {name}...")
        evaluator = Evaluator(model, class_names)
        metrics = evaluator.evaluate(X_test, y_test)
        results[name] = metrics

    # Create comparison summary
    summary = {
        'models': list(results.keys()),
        'accuracy': {name: m['accuracy'] for name, m in results.items()},
        'f1_score': {name: m['f1_score'] for name, m in results.items()},
        'precision': {name: m['precision'] for name, m in results.items()},
        'recall': {name: m['recall'] for name, m in results.items()},
    }

    # Find best model
    best_model = max(summary['accuracy'].items(), key=lambda x: x[1])
    summary['best_model'] = {
        'name': best_model[0],
        'accuracy': best_model[1]
    }

    results['summary'] = summary

    return results
