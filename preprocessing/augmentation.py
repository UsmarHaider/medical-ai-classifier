"""
Data Augmentation Module for Medical Images
Provides comprehensive augmentation strategies for different medical imaging modalities
"""
import numpy as np
import cv2
from typing import Tuple, Dict, List, Optional, Callable
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataAugmentor:
    """
    Data augmentation class for medical images.
    Provides modality-specific augmentation strategies.
    """

    def __init__(self, modality: str = "default"):
        """
        Initialize the data augmentor.

        Args:
            modality: Type of medical imaging modality
        """
        self.modality = modality
        self.augmentation_functions = self._get_augmentation_functions()

    def _get_augmentation_functions(self) -> Dict[str, Callable]:
        """Get modality-specific augmentation functions."""
        base_augmentations = {
            "horizontal_flip": self.horizontal_flip,
            "vertical_flip": self.vertical_flip,
            "rotation": self.rotation,
            "zoom": self.zoom,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "gaussian_noise": self.gaussian_noise,
            "elastic_transform": self.elastic_transform,
        }

        # Modality-specific configurations
        modality_configs = {
            "xray": ["horizontal_flip", "rotation", "brightness", "contrast", "gaussian_noise"],
            "ct": ["rotation", "zoom", "brightness", "contrast", "gaussian_noise"],
            "mri": ["rotation", "zoom", "brightness", "contrast", "gaussian_noise", "elastic_transform"],
            "microscopy": ["horizontal_flip", "vertical_flip", "rotation", "brightness", "contrast"],
            "skin": ["horizontal_flip", "vertical_flip", "rotation", "zoom", "brightness", "contrast"],
            "default": ["horizontal_flip", "rotation", "brightness", "contrast"]
        }

        enabled_augs = modality_configs.get(self.modality, modality_configs["default"])
        return {k: v for k, v in base_augmentations.items() if k in enabled_augs}

    def horizontal_flip(self, image: np.ndarray, p: float = 0.5) -> np.ndarray:
        """Apply horizontal flip with probability p."""
        if np.random.random() < p:
            return np.fliplr(image)
        return image

    def vertical_flip(self, image: np.ndarray, p: float = 0.5) -> np.ndarray:
        """Apply vertical flip with probability p."""
        if np.random.random() < p:
            return np.flipud(image)
        return image

    def rotation(self, image: np.ndarray,
                 angle_range: Tuple[int, int] = (-20, 20),
                 p: float = 0.5) -> np.ndarray:
        """Apply random rotation within angle range."""
        if np.random.random() < p:
            angle = np.random.uniform(angle_range[0], angle_range[1])
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        return image

    def zoom(self, image: np.ndarray,
             zoom_range: Tuple[float, float] = (0.8, 1.2),
             p: float = 0.5) -> np.ndarray:
        """Apply random zoom."""
        if np.random.random() < p:
            zoom_factor = np.random.uniform(zoom_range[0], zoom_range[1])
            h, w = image.shape[:2]
            new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)

            # Resize image
            resized = cv2.resize(image, (new_w, new_h))

            # Crop or pad to original size
            if zoom_factor > 1:
                # Crop center
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                return resized[start_h:start_h + h, start_w:start_w + w]
            else:
                # Pad
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                return cv2.copyMakeBorder(
                    resized, pad_h, h - new_h - pad_h,
                    pad_w, w - new_w - pad_w,
                    cv2.BORDER_REFLECT
                )
        return image

    def brightness(self, image: np.ndarray,
                   brightness_range: Tuple[float, float] = (0.8, 1.2),
                   p: float = 0.5) -> np.ndarray:
        """Apply random brightness adjustment."""
        if np.random.random() < p:
            factor = np.random.uniform(brightness_range[0], brightness_range[1])
            return np.clip(image * factor, 0, 1 if image.max() <= 1 else 255)
        return image

    def contrast(self, image: np.ndarray,
                 contrast_range: Tuple[float, float] = (0.8, 1.2),
                 p: float = 0.5) -> np.ndarray:
        """Apply random contrast adjustment."""
        if np.random.random() < p:
            factor = np.random.uniform(contrast_range[0], contrast_range[1])
            mean = image.mean()
            return np.clip((image - mean) * factor + mean, 0, 1 if image.max() <= 1 else 255)
        return image

    def gaussian_noise(self, image: np.ndarray,
                       noise_range: Tuple[float, float] = (0, 0.05),
                       p: float = 0.3) -> np.ndarray:
        """Add Gaussian noise to image."""
        if np.random.random() < p:
            sigma = np.random.uniform(noise_range[0], noise_range[1])
            noise = np.random.normal(0, sigma, image.shape)
            max_val = 1 if image.max() <= 1 else 255
            return np.clip(image + noise, 0, max_val)
        return image

    def elastic_transform(self, image: np.ndarray,
                          alpha: float = 50,
                          sigma: float = 5,
                          p: float = 0.3) -> np.ndarray:
        """Apply elastic transformation."""
        if np.random.random() < p:
            h, w = image.shape[:2]

            # Generate random displacement fields
            dx = cv2.GaussianBlur(
                (np.random.rand(h, w) * 2 - 1).astype(np.float32),
                (0, 0), sigma
            ) * alpha
            dy = cv2.GaussianBlur(
                (np.random.rand(h, w) * 2 - 1).astype(np.float32),
                (0, 0), sigma
            ) * alpha

            # Create mesh grid
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (x + dx).astype(np.float32)
            map_y = (y + dy).astype(np.float32)

            return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return image

    def augment(self, image: np.ndarray,
                augmentations: List[str] = None) -> np.ndarray:
        """
        Apply augmentations to an image.

        Args:
            image: Input image
            augmentations: List of augmentation names to apply (default: all enabled)

        Returns:
            Augmented image
        """
        if augmentations is None:
            augmentations = list(self.augmentation_functions.keys())

        result = image.copy()
        for aug_name in augmentations:
            if aug_name in self.augmentation_functions:
                result = self.augmentation_functions[aug_name](result)

        return result

    def augment_batch(self, images: np.ndarray,
                      num_augmentations: int = 1) -> np.ndarray:
        """
        Augment a batch of images.

        Args:
            images: Batch of images (N, H, W, C)
            num_augmentations: Number of augmented versions per image

        Returns:
            Augmented batch (N * num_augmentations, H, W, C)
        """
        augmented = []
        for image in images:
            for _ in range(num_augmentations):
                augmented.append(self.augment(image))

        return np.array(augmented)

    def get_keras_datagen(self,
                          training: bool = True) -> ImageDataGenerator:
        """
        Get Keras ImageDataGenerator with appropriate settings.

        Args:
            training: Whether this is for training (with augmentation)

        Returns:
            Configured ImageDataGenerator
        """
        if training:
            return ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True if self.modality not in ["xray"] else False,
                vertical_flip=True if self.modality in ["microscopy", "skin"] else False,
                fill_mode='nearest',
                brightness_range=[0.8, 1.2]
            )
        else:
            return ImageDataGenerator(rescale=1./255)

    def get_tf_augmentation_layer(self) -> tf.keras.Sequential:
        """
        Get TensorFlow augmentation layer for use in model.

        Returns:
            Sequential layer with augmentation
        """
        layers = [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ]

        if self.modality in ["microscopy", "skin"]:
            layers.append(tf.keras.layers.RandomFlip("vertical"))

        return tf.keras.Sequential(layers, name="augmentation")


def create_augmentor(modality: str = "default") -> DataAugmentor:
    """Factory function to create data augmentor."""
    return DataAugmentor(modality=modality)
