"""
Model Loader for Streamlit Application
Handles loading and caching of trained models
"""
import os
import sys
import numpy as np
from typing import Dict, Optional
import tensorflow as tf
from tensorflow import keras

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import DATASETS, MODELS_DIR


class ModelLoader:
    """
    Model loader class for loading and managing trained models.
    """

    def __init__(self):
        """Initialize the model loader."""
        self.models: Dict[str, keras.Model] = {}
        self.models_dir = MODELS_DIR

    def load_model(self, dataset_name: str) -> bool:
        """
        Load a trained model for a specific dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            True if model loaded successfully, False otherwise
        """
        if dataset_name in self.models:
            return True

        if dataset_name not in DATASETS:
            print(f"Unknown dataset: {dataset_name}")
            return False

        model_filename = DATASETS[dataset_name]["model_file"]
        model_path = os.path.join(self.models_dir, model_filename)

        # Try different file extensions
        possible_paths = [
            model_path,
            model_path.replace('.h5', '.keras'),
            model_path.replace('.h5', ''),  # SavedModel format
        ]

        for path in possible_paths:
            if os.path.exists(path):
                try:
                    print(f"Loading model from: {path}")
                    self.models[dataset_name] = keras.models.load_model(path)
                    print(f"Model loaded successfully for {dataset_name}")
                    return True
                except Exception as e:
                    print(f"Error loading model: {e}")
                    continue

        print(f"Model not found for {dataset_name}")
        return False

    def get_model(self, dataset_name: str) -> Optional[keras.Model]:
        """Get a loaded model."""
        if dataset_name not in self.models:
            self.load_model(dataset_name)
        return self.models.get(dataset_name)

    def predict(self,
                dataset_name: str,
                image: np.ndarray) -> Optional[Dict]:
        """
        Make a prediction on an image.

        Args:
            dataset_name: Name of the dataset
            image: Preprocessed image array

        Returns:
            Dictionary with prediction results
        """
        model = self.get_model(dataset_name)
        if model is None:
            return None

        # Ensure batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        # Make prediction
        probabilities = model.predict(image, verbose=0)

        # Handle different output formats
        if probabilities.shape[-1] == 1:
            # Binary classification with sigmoid
            prob = float(probabilities[0][0])
            probabilities = np.array([[1 - prob, prob]])

        probabilities = probabilities[0]

        # Get predicted class
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities.tolist()
        }

    def unload_model(self, dataset_name: str):
        """Unload a model to free memory."""
        if dataset_name in self.models:
            del self.models[dataset_name]
            # Clear Keras session
            keras.backend.clear_session()

    def unload_all(self):
        """Unload all models."""
        self.models.clear()
        keras.backend.clear_session()

    def get_loaded_models(self) -> list:
        """Get list of currently loaded models."""
        return list(self.models.keys())

    def get_model_info(self, dataset_name: str) -> Optional[Dict]:
        """Get information about a model."""
        model = self.get_model(dataset_name)
        if model is None:
            return None

        return {
            "name": model.name,
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
            "num_parameters": model.count_params(),
            "num_layers": len(model.layers)
        }


class EnsemblePredictor:
    """
    Ensemble predictor that combines predictions from multiple models.
    """

    def __init__(self, model_loader: ModelLoader):
        """Initialize ensemble predictor."""
        self.model_loader = model_loader

    def predict_ensemble(self,
                         dataset_name: str,
                         image: np.ndarray,
                         model_weights: Dict[str, float] = None) -> Optional[Dict]:
        """
        Make ensemble prediction using multiple model types.

        Args:
            dataset_name: Name of the dataset
            image: Preprocessed image array
            model_weights: Weights for each model type

        Returns:
            Dictionary with ensemble prediction results
        """
        # Model variants to try
        model_variants = [
            f"{dataset_name}_custom_cnn_model.h5",
            f"{dataset_name}_vgg16_model.h5",
            f"{dataset_name}_resnet50_model.h5"
        ]

        predictions = []
        weights = []

        for variant in model_variants:
            model_path = os.path.join(MODELS_DIR, variant)
            if os.path.exists(model_path):
                try:
                    model = keras.models.load_model(model_path)
                    pred = model.predict(np.expand_dims(image, 0), verbose=0)
                    predictions.append(pred[0])
                    weights.append(model_weights.get(variant, 1.0) if model_weights else 1.0)
                except Exception as e:
                    print(f"Error with {variant}: {e}")

        if not predictions:
            # Fall back to single model
            return self.model_loader.predict(dataset_name, image)

        # Weighted average of predictions
        weights = np.array(weights) / np.sum(weights)
        ensemble_probs = np.average(predictions, axis=0, weights=weights)

        # Get predicted class
        if len(ensemble_probs.shape) == 1:
            if len(ensemble_probs) == 1:
                prob = float(ensemble_probs[0])
                ensemble_probs = np.array([1 - prob, prob])

        predicted_class = int(np.argmax(ensemble_probs))
        confidence = float(ensemble_probs[predicted_class])

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": ensemble_probs.tolist(),
            "num_models": len(predictions)
        }


def get_available_models() -> Dict[str, bool]:
    """
    Check which models are available.

    Returns:
        Dictionary of dataset_name -> model_available
    """
    available = {}
    for dataset_name, config in DATASETS.items():
        model_path = os.path.join(MODELS_DIR, config["model_file"])
        available[dataset_name] = os.path.exists(model_path)
    return available
