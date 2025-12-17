"""
Model Factory for Medical Image Classification
Provides unified interface for creating different model architectures
"""
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, Optional, Dict
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .custom_cnn import CustomCNN, create_custom_cnn
from .transfer_learning import (
    VGG16Model, ResNet50Model, InceptionV3Model,
    create_vgg16, create_resnet50, create_inceptionv3
)
from .vision_transformer import VisionTransformer, create_vit


class ModelFactory:
    """
    Factory class for creating medical image classification models.
    """

    AVAILABLE_MODELS = {
        "custom_cnn": {
            "name": "Custom CNN",
            "description": "Custom Convolutional Neural Network",
            "variants": ["standard", "deep", "lightweight"]
        },
        "vgg16": {
            "name": "VGG16",
            "description": "VGG16 with ImageNet pre-trained weights",
            "variants": ["frozen", "fine_tune"]
        },
        "resnet50": {
            "name": "ResNet50",
            "description": "ResNet50 with ImageNet pre-trained weights",
            "variants": ["frozen", "fine_tune"]
        },
        "inceptionv3": {
            "name": "InceptionV3",
            "description": "InceptionV3 with ImageNet pre-trained weights",
            "variants": ["frozen", "fine_tune"]
        },
        "vit": {
            "name": "Vision Transformer",
            "description": "Vision Transformer (ViT)",
            "variants": ["tiny", "small", "base"]
        }
    }

    def __init__(self,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 2):
        """
        Initialize the model factory.

        Args:
            input_shape: Input image shape
            num_classes: Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes

    def create_model(self,
                     model_type: str,
                     variant: str = None,
                     **kwargs) -> keras.Model:
        """
        Create a model of the specified type.

        Args:
            model_type: Type of model to create
            variant: Model variant (optional)
            **kwargs: Additional arguments for model creation

        Returns:
            Compiled Keras model
        """
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available: {list(self.AVAILABLE_MODELS.keys())}")

        if model_type == "custom_cnn":
            return self._create_custom_cnn(variant or "standard", **kwargs)
        elif model_type == "vgg16":
            return self._create_vgg16(variant or "frozen", **kwargs)
        elif model_type == "resnet50":
            return self._create_resnet50(variant or "frozen", **kwargs)
        elif model_type == "inceptionv3":
            return self._create_inceptionv3(variant or "frozen", **kwargs)
        elif model_type == "vit":
            return self._create_vit(variant or "small", **kwargs)

    def _create_custom_cnn(self, variant: str, **kwargs) -> keras.Model:
        """Create Custom CNN model."""
        cnn = CustomCNN(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            architecture=variant
        )
        model = cnn.get_model()
        return model

    def _create_vgg16(self, variant: str, **kwargs) -> keras.Model:
        """Create VGG16 model."""
        freeze_base = variant == "frozen"
        model = VGG16Model(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            freeze_base=freeze_base,
            fine_tune_layers=kwargs.get('fine_tune_layers', 0)
        )
        return model.get_model()

    def _create_resnet50(self, variant: str, **kwargs) -> keras.Model:
        """Create ResNet50 model."""
        freeze_base = variant == "frozen"
        model = ResNet50Model(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            freeze_base=freeze_base,
            fine_tune_layers=kwargs.get('fine_tune_layers', 0)
        )
        return model.get_model()

    def _create_inceptionv3(self, variant: str, **kwargs) -> keras.Model:
        """Create InceptionV3 model."""
        # InceptionV3 requires minimum 139x139 input
        input_shape = self.input_shape
        if input_shape[0] < 139 or input_shape[1] < 139:
            input_shape = (299, 299, 3)

        freeze_base = variant == "frozen"
        model = InceptionV3Model(
            input_shape=input_shape,
            num_classes=self.num_classes,
            freeze_base=freeze_base,
            fine_tune_layers=kwargs.get('fine_tune_layers', 0)
        )
        return model.get_model()

    def _create_vit(self, variant: str, **kwargs) -> keras.Model:
        """Create Vision Transformer model."""
        vit = VisionTransformer(
            input_shape=self.input_shape,
            num_classes=self.num_classes
        )

        if variant == "tiny":
            vit.build_tiny()
        elif variant == "small":
            vit.build_small()
        else:
            vit._build_functional()

        vit.compile()
        return vit.model

    def create_ensemble(self,
                        model_types: list,
                        weights: list = None) -> keras.Model:
        """
        Create an ensemble of models.

        Args:
            model_types: List of model types to include
            weights: Weights for each model (optional)

        Returns:
            Ensemble model
        """
        models = [self.create_model(mt) for mt in model_types]

        if weights is None:
            weights = [1.0 / len(models)] * len(models)

        # Create ensemble
        inputs = keras.Input(shape=self.input_shape)
        outputs = []

        for model in models:
            # Clone model to avoid weight sharing issues
            model_clone = keras.models.clone_model(model)
            output = model_clone(inputs)
            outputs.append(output)

        # Average predictions
        if len(outputs) > 1:
            averaged = keras.layers.Average()(outputs)
        else:
            averaged = outputs[0]

        ensemble = keras.Model(inputs=inputs, outputs=averaged, name='ensemble')

        # Compile
        if self.num_classes == 2:
            ensemble.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            ensemble.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

        return ensemble

    @staticmethod
    def list_models() -> Dict:
        """List all available models."""
        return ModelFactory.AVAILABLE_MODELS

    @staticmethod
    def get_model_info(model_type: str) -> Dict:
        """Get information about a specific model type."""
        if model_type not in ModelFactory.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model type: {model_type}")
        return ModelFactory.AVAILABLE_MODELS[model_type]


def create_model(model_type: str,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 2,
                 variant: str = None,
                 **kwargs) -> keras.Model:
    """
    Convenience function to create a model.

    Args:
        model_type: Type of model to create
        input_shape: Input image shape
        num_classes: Number of output classes
        variant: Model variant
        **kwargs: Additional arguments

    Returns:
        Compiled Keras model
    """
    factory = ModelFactory(input_shape=input_shape, num_classes=num_classes)
    return factory.create_model(model_type, variant, **kwargs)
