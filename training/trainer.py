"""
Training Module for Medical Image Classification Models
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Tuple, Optional, List
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import DATASETS, TRAINING_CONFIG, MODELS_DIR
from preprocessing.data_loader import DataLoader
from models.model_factory import ModelFactory
from .callbacks import get_callbacks


class Trainer:
    """
    Trainer class for medical image classification models.
    """

    def __init__(self,
                 dataset_name: str,
                 model_type: str = "custom_cnn",
                 model_variant: str = None,
                 data_dir: str = None):
        """
        Initialize the trainer.

        Args:
            dataset_name: Name of the dataset
            model_type: Type of model to train
            model_variant: Model variant (optional)
            data_dir: Path to data directory
        """
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        self.dataset_name = dataset_name
        self.model_type = model_type
        self.model_variant = model_variant
        self.data_dir = data_dir

        # Get dataset configuration
        self.dataset_config = DATASETS[dataset_name]
        self.image_size = self.dataset_config["image_size"]
        self.num_classes = self.dataset_config["num_classes"]

        # Initialize data loader
        self.data_loader = DataLoader(dataset_name=dataset_name, data_dir=data_dir)

        # Initialize model factory
        self.model_factory = ModelFactory(
            input_shape=(*self.image_size, 3),
            num_classes=self.num_classes
        )

        # Model and history
        self.model = None
        self.history = None

    def build_model(self) -> keras.Model:
        """Build the model."""
        self.model = self.model_factory.create_model(
            model_type=self.model_type,
            variant=self.model_variant
        )
        return self.model

    def train(self,
              epochs: int = None,
              batch_size: int = None,
              use_data_generators: bool = True,
              class_weights: bool = True,
              callbacks: List = None,
              verbose: int = 1) -> Dict:
        """
        Train the model.

        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            use_data_generators: Whether to use Keras data generators
            class_weights: Whether to use class weights for imbalanced data
            callbacks: List of callbacks
            verbose: Verbosity level

        Returns:
            Training history
        """
        epochs = epochs or TRAINING_CONFIG["epochs"]
        batch_size = batch_size or TRAINING_CONFIG["batch_size"]

        # Build model if not already built
        if self.model is None:
            self.build_model()

        # Get callbacks
        if callbacks is None:
            callbacks = get_callbacks(
                self.dataset_name,
                self.model_type,
                patience=TRAINING_CONFIG["early_stopping_patience"]
            )

        if use_data_generators:
            return self._train_with_generators(epochs, batch_size, class_weights, callbacks, verbose)
        else:
            return self._train_with_numpy(epochs, batch_size, class_weights, callbacks, verbose)

    def _train_with_generators(self,
                               epochs: int,
                               batch_size: int,
                               class_weights: bool,
                               callbacks: List,
                               verbose: int) -> Dict:
        """Train using Keras data generators."""
        # Create data generators
        generators = self.data_loader.create_data_generators()

        if generators is None:
            print("Failed to create data generators. Check data directory.")
            return None

        train_generator = generators["train_generator"]
        val_generator = generators["val_generator"]

        # Calculate class weights if needed
        weights = None
        if class_weights:
            weights = self._calculate_class_weights(train_generator.classes)

        # Train model
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            class_weight=weights,
            verbose=verbose
        )

        return self.history.history

    def _train_with_numpy(self,
                          epochs: int,
                          batch_size: int,
                          class_weights: bool,
                          callbacks: List,
                          verbose: int) -> Dict:
        """Train using numpy arrays."""
        # Load data
        data = self.data_loader.load_numpy_data()

        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]

        # Calculate class weights if needed
        weights = None
        if class_weights:
            if len(y_train.shape) > 1:
                labels = np.argmax(y_train, axis=1)
            else:
                labels = y_train
            weights = self._calculate_class_weights(labels)

        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=weights,
            verbose=verbose
        )

        return self.history.history

    def _calculate_class_weights(self, labels: np.ndarray) -> Dict:
        """Calculate class weights for imbalanced data."""
        from sklearn.utils.class_weight import compute_class_weight

        unique_classes = np.unique(labels)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=labels
        )

        return {i: w for i, w in enumerate(weights)}

    def evaluate(self, test_data: Tuple = None) -> Dict:
        """
        Evaluate the model on test data.

        Args:
            test_data: Optional tuple of (X_test, y_test)

        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        if test_data is None:
            # Load test data
            data = self.data_loader.load_numpy_data()
            X_test, y_test = data["X_test"], data["y_test"]
        else:
            X_test, y_test = test_data

        # Evaluate
        results = self.model.evaluate(X_test, y_test, verbose=0)

        # Get metric names
        metric_names = self.model.metrics_names

        return {name: value for name, value in zip(metric_names, results)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input data."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        return self.model.predict(X)

    def save_model(self, path: str = None):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        if path is None:
            model_filename = self.dataset_config["model_file"]
            path = os.path.join(MODELS_DIR, model_filename)

        # Create directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save model
        self.model.save(path)
        print(f"Model saved to: {path}")

    def load_model(self, path: str = None):
        """Load a trained model."""
        if path is None:
            model_filename = self.dataset_config["model_file"]
            path = os.path.join(MODELS_DIR, model_filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        self.model = keras.models.load_model(path)
        print(f"Model loaded from: {path}")

    def save_training_history(self, path: str = None):
        """Save training history to JSON."""
        if self.history is None:
            raise ValueError("No training history to save")

        if path is None:
            path = os.path.join(
                MODELS_DIR,
                f"{self.dataset_name}_{self.model_type}_history.json"
            )

        # Convert numpy types to Python types
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(v) for v in values]

        with open(path, 'w') as f:
            json.dump(history_dict, f, indent=2)

        print(f"Training history saved to: {path}")

    def fine_tune(self,
                  epochs: int = 10,
                  learning_rate: float = 0.00001,
                  layers_to_unfreeze: int = 20):
        """
        Fine-tune the model by unfreezing some layers.

        Args:
            epochs: Number of fine-tuning epochs
            learning_rate: Learning rate for fine-tuning
            layers_to_unfreeze: Number of layers to unfreeze
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Unfreeze top layers
        for layer in self.model.layers[-layers_to_unfreeze:]:
            layer.trainable = True

        # Recompile with lower learning rate
        if self.num_classes == 2:
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy', keras.metrics.AUC(name='auc')]
            )
        else:
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

        # Continue training
        callbacks = get_callbacks(
            self.dataset_name,
            f"{self.model_type}_finetune"
        )

        generators = self.data_loader.create_data_generators()
        if generators:
            history = self.model.fit(
                generators["train_generator"],
                epochs=epochs,
                validation_data=generators["val_generator"],
                callbacks=callbacks,
                verbose=1
            )
            return history.history

        return None


def train_all_models(dataset_name: str,
                     model_types: List[str] = None,
                     epochs: int = 50,
                     save_models: bool = True) -> Dict:
    """
    Train all model architectures on a dataset.

    Args:
        dataset_name: Name of the dataset
        model_types: List of model types to train (default: all)
        epochs: Number of epochs
        save_models: Whether to save trained models

    Returns:
        Dictionary of training results
    """
    if model_types is None:
        model_types = ["custom_cnn", "vgg16", "resnet50", "vit"]

    results = {}

    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Training {model_type} on {dataset_name}")
        print('='*50)

        try:
            trainer = Trainer(
                dataset_name=dataset_name,
                model_type=model_type
            )

            history = trainer.train(epochs=epochs)

            if history:
                evaluation = trainer.evaluate()
                results[model_type] = {
                    "history": history,
                    "evaluation": evaluation
                }

                if save_models:
                    # Save with model type in filename
                    model_path = os.path.join(
                        MODELS_DIR,
                        f"{dataset_name}_{model_type}_model.h5"
                    )
                    trainer.save_model(model_path)

        except Exception as e:
            print(f"Error training {model_type}: {e}")
            results[model_type] = {"error": str(e)}

    return results
