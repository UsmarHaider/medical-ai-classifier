"""
Bone Shadow Suppression Module for Chest X-Ray Images
Implements various techniques for suppressing bone shadows in chest radiographs
"""
import numpy as np
import cv2
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class BoneShadowSuppressor:
    """
    Bone shadow suppression for chest X-ray images.
    Uses both traditional image processing and deep learning approaches.
    """

    def __init__(self, method: str = "unet"):
        """
        Initialize the bone shadow suppressor.

        Args:
            method: Suppression method ('unet', 'traditional', 'hybrid')
        """
        self.method = method
        self.model = None
        self.image_size = (256, 256)

    def traditional_suppression(self, image: np.ndarray) -> np.ndarray:
        """
        Apply traditional image processing for bone shadow suppression.

        Uses multi-scale decomposition and selective filtering.

        Args:
            image: Input chest X-ray image

        Returns:
            Image with suppressed bone shadows
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Normalize to float
        gray = gray.astype(np.float32) / 255.0

        # Multi-scale decomposition using Gaussian pyramids
        # Level 1: Original resolution
        g1 = gray

        # Level 2: Half resolution
        g2 = cv2.pyrDown(g1)
        g2_up = cv2.pyrUp(g2, dstsize=(g1.shape[1], g1.shape[0]))

        # Level 3: Quarter resolution
        g3 = cv2.pyrDown(g2)
        g3_up = cv2.pyrUp(g3, dstsize=(g2.shape[1], g2.shape[0]))
        g3_up = cv2.pyrUp(g3_up, dstsize=(g1.shape[1], g1.shape[0]))

        # Laplacian pyramid (detail layers)
        l1 = g1 - g2_up  # High frequency details
        l2 = g2_up - g3_up  # Medium frequency details

        # Bone shadows tend to have medium-frequency characteristics
        # Suppress medium frequencies selectively
        l2_suppressed = l2 * 0.5

        # Reconstruct image
        reconstructed = g3_up + l2_suppressed + l1

        # Enhance soft tissue visibility
        # Apply CLAHE
        reconstructed_uint8 = (np.clip(reconstructed, 0, 1) * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(reconstructed_uint8)

        # Convert back to 3 channels if input was RGB
        if len(image.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

        return enhanced

    def build_unet_model(self, input_shape: Tuple[int, int, int] = (256, 256, 1)) -> keras.Model:
        """
        Build U-Net model for bone shadow suppression.

        Args:
            input_shape: Shape of input images

        Returns:
            Compiled U-Net model
        """
        inputs = layers.Input(shape=input_shape)

        # Encoder
        # Block 1
        c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
        c1 = layers.BatchNormalization()(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        # Block 2
        c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
        c2 = layers.BatchNormalization()(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        # Block 3
        c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
        c3 = layers.BatchNormalization()(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        # Block 4
        c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
        c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
        c4 = layers.BatchNormalization()(c4)
        p4 = layers.MaxPooling2D((2, 2))(c4)

        # Bridge
        c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
        c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
        c5 = layers.BatchNormalization()(c5)

        # Decoder
        # Block 6
        u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = layers.concatenate([u6, c4])
        c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
        c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
        c6 = layers.BatchNormalization()(c6)

        # Block 7
        u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c3])
        c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
        c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
        c7 = layers.BatchNormalization()(c7)

        # Block 8
        u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = layers.concatenate([u8, c2])
        c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
        c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
        c8 = layers.BatchNormalization()(c8)

        # Block 9
        u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = layers.concatenate([u9, c1])
        c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
        c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
        c9 = layers.BatchNormalization()(c9)

        # Output
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

        model = keras.Model(inputs=[inputs], outputs=[outputs])
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

        self.model = model
        return model

    def load_model(self, model_path: str):
        """Load pre-trained model."""
        self.model = keras.models.load_model(model_path)

    def save_model(self, model_path: str):
        """Save trained model."""
        if self.model:
            self.model.save(model_path)

    def deep_learning_suppression(self, image: np.ndarray) -> np.ndarray:
        """
        Apply deep learning-based bone shadow suppression.

        Args:
            image: Input chest X-ray image

        Returns:
            Image with suppressed bone shadows
        """
        if self.model is None:
            self.build_unet_model()

        # Preprocess image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Resize to model input size
        original_size = gray.shape[:2]
        resized = cv2.resize(gray, self.image_size)

        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        input_tensor = normalized.reshape(1, *self.image_size, 1)

        # Predict
        output = self.model.predict(input_tensor, verbose=0)
        output = output.reshape(self.image_size)

        # Resize back to original size
        result = cv2.resize(output, (original_size[1], original_size[0]))

        # Convert back to uint8
        result = (result * 255).astype(np.uint8)

        # Convert to RGB if input was RGB
        if len(image.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

        return result

    def hybrid_suppression(self, image: np.ndarray) -> np.ndarray:
        """
        Combine traditional and deep learning approaches.

        Args:
            image: Input chest X-ray image

        Returns:
            Image with suppressed bone shadows
        """
        # Apply traditional method first
        traditional_result = self.traditional_suppression(image)

        # Apply deep learning refinement
        if self.model is not None:
            # Use traditional result as input to deep learning model
            dl_result = self.deep_learning_suppression(traditional_result)
            # Blend results
            alpha = 0.6  # Weight for deep learning result
            result = cv2.addWeighted(dl_result, alpha, traditional_result, 1 - alpha, 0)
        else:
            result = traditional_result

        return result

    def suppress(self, image: np.ndarray) -> np.ndarray:
        """
        Apply bone shadow suppression using the configured method.

        Args:
            image: Input chest X-ray image

        Returns:
            Image with suppressed bone shadows
        """
        if self.method == "traditional":
            return self.traditional_suppression(image)
        elif self.method == "unet":
            return self.deep_learning_suppression(image)
        elif self.method == "hybrid":
            return self.hybrid_suppression(image)
        else:
            return self.traditional_suppression(image)

    def train(self,
              X_original: np.ndarray,
              X_suppressed: np.ndarray,
              epochs: int = 50,
              batch_size: int = 8,
              validation_split: float = 0.1):
        """
        Train the U-Net model for bone shadow suppression.

        Args:
            X_original: Original X-ray images with bone shadows
            X_suppressed: Ground truth images without bone shadows
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Validation data fraction
        """
        if self.model is None:
            self.build_unet_model()

        # Ensure correct shape
        if len(X_original.shape) == 3:
            X_original = np.expand_dims(X_original, -1)
        if len(X_suppressed.shape) == 3:
            X_suppressed = np.expand_dims(X_suppressed, -1)

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
        ]

        # Train
        history = self.model.fit(
            X_original, X_suppressed,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        return history


def create_bone_shadow_suppressor(method: str = "traditional") -> BoneShadowSuppressor:
    """Factory function to create bone shadow suppressor."""
    return BoneShadowSuppressor(method=method)
