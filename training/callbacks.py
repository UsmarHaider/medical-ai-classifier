"""
Training Callbacks for Medical Image Classification
"""
import os
import sys
from tensorflow import keras
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import MODELS_DIR, TRAINING_CONFIG


def get_callbacks(dataset_name: str,
                  model_type: str,
                  patience: int = None,
                  monitor: str = 'val_loss',
                  save_best_only: bool = True) -> List[keras.callbacks.Callback]:
    """
    Get training callbacks.

    Args:
        dataset_name: Name of the dataset
        model_type: Type of model being trained
        patience: Early stopping patience
        monitor: Metric to monitor
        save_best_only: Whether to save only best model

    Returns:
        List of callbacks
    """
    patience = patience or TRAINING_CONFIG["early_stopping_patience"]

    callbacks = []

    # Early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)

    # Reduce learning rate on plateau
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=0.5,
        patience=TRAINING_CONFIG.get("reduce_lr_patience", 5),
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(reduce_lr)

    # Model checkpoint
    checkpoint_path = os.path.join(
        MODELS_DIR,
        f"{dataset_name}_{model_type}_best.h5"
    )
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=monitor,
        save_best_only=save_best_only,
        save_weights_only=False,
        verbose=1
    )
    callbacks.append(model_checkpoint)

    # TensorBoard logging
    log_dir = os.path.join(
        os.path.dirname(MODELS_DIR),
        "logs",
        f"{dataset_name}_{model_type}"
    )
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True
    )
    callbacks.append(tensorboard)

    # CSV Logger
    csv_path = os.path.join(
        MODELS_DIR,
        f"{dataset_name}_{model_type}_training_log.csv"
    )
    csv_logger = keras.callbacks.CSVLogger(csv_path)
    callbacks.append(csv_logger)

    return callbacks


class CustomCallback(keras.callbacks.Callback):
    """Custom callback for additional monitoring."""

    def __init__(self, dataset_name: str, model_type: str):
        super().__init__()
        self.dataset_name = dataset_name
        self.model_type = model_type

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        logs = logs or {}

        # Log progress
        val_acc = logs.get('val_accuracy', 0)
        val_loss = logs.get('val_loss', 0)

        print(f"\n[{self.dataset_name}][{self.model_type}] "
              f"Epoch {epoch + 1}: val_accuracy={val_acc:.4f}, val_loss={val_loss:.4f}")

    def on_train_end(self, logs=None):
        """Called at the end of training."""
        print(f"\n[{self.dataset_name}][{self.model_type}] Training completed!")


class MetricsLogger(keras.callbacks.Callback):
    """Callback to log detailed metrics."""

    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.train_losses.append(logs.get('loss', 0))
        self.val_losses.append(logs.get('val_loss', 0))
        self.train_accs.append(logs.get('accuracy', 0))
        self.val_accs.append(logs.get('val_accuracy', 0))

    def get_metrics(self):
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accs,
            'val_accuracies': self.val_accs
        }


def get_callbacks_for_fine_tuning(dataset_name: str,
                                   model_type: str) -> List[keras.callbacks.Callback]:
    """Get callbacks specifically for fine-tuning phase."""
    callbacks = []

    # More conservative early stopping for fine-tuning
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)

    # Smaller learning rate reduction
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=7,
        min_lr=1e-8,
        verbose=1
    )
    callbacks.append(reduce_lr)

    # Model checkpoint for fine-tuned model
    checkpoint_path = os.path.join(
        MODELS_DIR,
        f"{dataset_name}_{model_type}_finetuned_best.h5"
    )
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    callbacks.append(model_checkpoint)

    return callbacks
