"""
Image Preprocessing Module for Medical Images
Handles modality-specific image normalization and preprocessing
"""
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from typing import Tuple, Optional, Union
import os


class ImagePreprocessor:
    """
    Comprehensive image preprocessor for medical images.
    Supports different modalities: X-Ray, CT, MRI, Microscopy, Skin Images
    """

    MODALITY_CONFIGS = {
        "xray": {
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid_size": (8, 8),
            "normalize_method": "minmax",
            "denoise": True
        },
        "ct": {
            "window_center": 40,
            "window_width": 400,
            "normalize_method": "window",
            "denoise": True
        },
        "mri": {
            "bias_correction": True,
            "normalize_method": "zscore",
            "denoise": True
        },
        "microscopy": {
            "color_normalization": True,
            "normalize_method": "stain",
            "denoise": False
        },
        "skin": {
            "hair_removal": False,
            "normalize_method": "minmax",
            "denoise": False
        },
        "default": {
            "normalize_method": "minmax",
            "denoise": False
        }
    }

    # Dataset to modality mapping
    DATASET_MODALITY = {
        "kidney_cancer": "ct",
        "cervical_cancer": "microscopy",
        "alzheimer": "mri",
        "covid19": "xray",
        "pneumonia": "xray",
        "tuberculosis": "xray",
        "monkeypox": "skin",
        "malaria": "microscopy",
        "bone_shadow": "xray"
    }

    def __init__(self, dataset_name: str = None, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the image preprocessor.

        Args:
            dataset_name: Name of the dataset to determine modality
            target_size: Target size for resizing images
        """
        self.target_size = target_size
        self.dataset_name = dataset_name

        # Determine modality based on dataset
        if dataset_name and dataset_name in self.DATASET_MODALITY:
            self.modality = self.DATASET_MODALITY[dataset_name]
        else:
            self.modality = "default"

        self.config = self.MODALITY_CONFIGS[self.modality]

    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from file path."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            # Try loading with PIL for more format support
            pil_image = Image.open(image_path)
            image = np.array(pil_image)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size."""
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(
                clipLimit=self.config.get("clahe_clip_limit", 2.0),
                tileGridSize=self.config.get("clahe_tile_grid_size", (8, 8))
            )
            l = clahe.apply(l)

            # Merge channels
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)

    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising to image."""
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

    def normalize_minmax(self, image: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0, 1] range."""
        image = image.astype(np.float32)
        min_val = image.min()
        max_val = image.max()
        if max_val - min_val > 0:
            image = (image - min_val) / (max_val - min_val)
        return image

    def normalize_zscore(self, image: np.ndarray) -> np.ndarray:
        """Z-score normalization (standardization)."""
        image = image.astype(np.float32)
        mean = image.mean()
        std = image.std()
        if std > 0:
            image = (image - mean) / std
        # Clip to reasonable range and rescale to [0, 1]
        image = np.clip(image, -3, 3)
        image = (image + 3) / 6
        return image

    def normalize_window(self, image: np.ndarray) -> np.ndarray:
        """Window-level normalization for CT images."""
        center = self.config.get("window_center", 40)
        width = self.config.get("window_width", 400)

        image = image.astype(np.float32)
        min_val = center - width / 2
        max_val = center + width / 2

        image = np.clip(image, min_val, max_val)
        image = (image - min_val) / (max_val - min_val)
        return image

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Apply appropriate normalization based on modality."""
        method = self.config.get("normalize_method", "minmax")

        if method == "minmax":
            return self.normalize_minmax(image)
        elif method == "zscore":
            return self.normalize_zscore(image)
        elif method == "window":
            return self.normalize_window(image)
        elif method == "stain":
            # For microscopy, use simple min-max for now
            return self.normalize_minmax(image)
        else:
            return self.normalize_minmax(image)

    def preprocess_xray(self, image: np.ndarray) -> np.ndarray:
        """Specific preprocessing for X-ray images."""
        # Apply CLAHE for contrast enhancement
        image = self.apply_clahe(image)

        # Denoise if configured
        if self.config.get("denoise", False):
            image = self.denoise_image(image)

        return image

    def preprocess_ct(self, image: np.ndarray) -> np.ndarray:
        """Specific preprocessing for CT images."""
        # Apply window-level normalization
        if self.config.get("denoise", False):
            image = self.denoise_image(image)
        return image

    def preprocess_mri(self, image: np.ndarray) -> np.ndarray:
        """Specific preprocessing for MRI images."""
        # Apply denoising
        if self.config.get("denoise", False):
            image = self.denoise_image(image)
        return image

    def preprocess_microscopy(self, image: np.ndarray) -> np.ndarray:
        """Specific preprocessing for microscopy images."""
        # Color normalization can be added here
        return image

    def preprocess_skin(self, image: np.ndarray) -> np.ndarray:
        """Specific preprocessing for skin images."""
        return image

    def preprocess(self, image: Union[str, np.ndarray],
                   expand_dims: bool = False) -> np.ndarray:
        """
        Main preprocessing pipeline.

        Args:
            image: Image path or numpy array
            expand_dims: Whether to add batch dimension

        Returns:
            Preprocessed image as numpy array
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = self.load_image(image)

        # Resize image
        image = self.resize_image(image)

        # Apply modality-specific preprocessing
        if self.modality == "xray":
            image = self.preprocess_xray(image)
        elif self.modality == "ct":
            image = self.preprocess_ct(image)
        elif self.modality == "mri":
            image = self.preprocess_mri(image)
        elif self.modality == "microscopy":
            image = self.preprocess_microscopy(image)
        elif self.modality == "skin":
            image = self.preprocess_skin(image)

        # Normalize image
        image = self.normalize_image(image)

        # Ensure 3 channels
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 1:
            image = np.concatenate([image] * 3, axis=-1)

        # Add batch dimension if requested
        if expand_dims:
            image = np.expand_dims(image, axis=0)

        return image.astype(np.float32)

    def preprocess_batch(self, images: list,
                         show_progress: bool = False) -> np.ndarray:
        """
        Preprocess a batch of images.

        Args:
            images: List of image paths or numpy arrays
            show_progress: Whether to show progress bar

        Returns:
            Batch of preprocessed images
        """
        if show_progress:
            from tqdm import tqdm
            images = tqdm(images, desc="Preprocessing images")

        processed = [self.preprocess(img) for img in images]
        return np.array(processed)


def create_preprocessor(dataset_name: str,
                        target_size: Tuple[int, int] = (224, 224)) -> ImagePreprocessor:
    """Factory function to create preprocessor for specific dataset."""
    return ImagePreprocessor(dataset_name=dataset_name, target_size=target_size)
