"""
Image Processing Module for Streamlit Application
"""
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Union
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.image_preprocessor import ImagePreprocessor


def process_image(image: Union[Image.Image, np.ndarray, str],
                  dataset_name: str = None,
                  target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Process an image for model prediction.

    Args:
        image: PIL Image, numpy array, or file path
        dataset_name: Name of the dataset for modality-specific processing
        target_size: Target size for the image

    Returns:
        Preprocessed image as numpy array
    """
    # Convert PIL Image to numpy array
    if isinstance(image, Image.Image):
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image)

    # Load from path if string
    elif isinstance(image, str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize preprocessor
    preprocessor = ImagePreprocessor(
        dataset_name=dataset_name,
        target_size=target_size
    )

    # Process image
    processed = preprocessor.preprocess(image, expand_dims=False)

    return processed


def load_and_process_image(file_path: str,
                           dataset_name: str = None,
                           target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Load and process an image from file path.

    Args:
        file_path: Path to the image file
        dataset_name: Name of the dataset
        target_size: Target size for the image

    Returns:
        Preprocessed image as numpy array
    """
    image = Image.open(file_path)
    return process_image(image, dataset_name, target_size)


def process_uploaded_file(uploaded_file,
                          dataset_name: str = None,
                          target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Process an uploaded file from Streamlit.

    Args:
        uploaded_file: Streamlit uploaded file object
        dataset_name: Name of the dataset
        target_size: Target size for the image

    Returns:
        Preprocessed image as numpy array
    """
    image = Image.open(uploaded_file)
    return process_image(image, dataset_name, target_size)


def validate_image(image: Union[Image.Image, np.ndarray]) -> Tuple[bool, str]:
    """
    Validate an image for processing.

    Args:
        image: Image to validate

    Returns:
        Tuple of (is_valid, message)
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Check dimensions
    if len(image.shape) < 2:
        return False, "Invalid image dimensions"

    # Check size
    if image.shape[0] < 32 or image.shape[1] < 32:
        return False, "Image too small. Minimum size is 32x32 pixels."

    if image.shape[0] > 4096 or image.shape[1] > 4096:
        return False, "Image too large. Maximum size is 4096x4096 pixels."

    # Check channels
    if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
        return False, "Invalid number of channels. Expected 1, 3, or 4."

    return True, "Image is valid"


def get_image_info(image: Union[Image.Image, np.ndarray]) -> dict:
    """
    Get information about an image.

    Args:
        image: Image to analyze

    Returns:
        Dictionary with image information
    """
    if isinstance(image, Image.Image):
        return {
            "format": image.format,
            "mode": image.mode,
            "size": image.size,
            "width": image.width,
            "height": image.height
        }
    else:
        return {
            "shape": image.shape,
            "dtype": str(image.dtype),
            "min_value": float(image.min()),
            "max_value": float(image.max()),
            "mean_value": float(image.mean())
        }


def resize_image_for_display(image: Union[Image.Image, np.ndarray],
                              max_size: int = 400) -> Image.Image:
    """
    Resize image for display in Streamlit.

    Args:
        image: Image to resize
        max_size: Maximum dimension size

    Returns:
        Resized PIL Image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Calculate new size maintaining aspect ratio
    width, height = image.size
    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def apply_gradcam(model,
                  image: np.ndarray,
                  layer_name: str = None) -> np.ndarray:
    """
    Apply GradCAM visualization to understand model predictions.

    Args:
        model: Keras model
        image: Input image
        layer_name: Name of convolutional layer (defaults to last conv layer)

    Returns:
        GradCAM heatmap overlaid on image
    """
    import tensorflow as tf

    # Find last convolutional layer if not specified
    if layer_name is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                layer_name = layer.name
                break

    if layer_name is None:
        return image

    # Create gradient model
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    # Ensure image has batch dimension
    if len(image.shape) == 3:
        image = np.expand_dims(image, 0)

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        if predictions.shape[-1] == 1:
            loss = predictions[:, 0]
        else:
            predicted_class = tf.argmax(predictions[0])
            loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)

    # Compute guided gradients
    guided_grads = grads[0]
    conv_outputs = conv_outputs[0]

    # Compute weights
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))

    # Compute CAM
    cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]

    # Normalize
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)

    # Resize to image size
    cam = cv2.resize(cam, (image.shape[2], image.shape[1]))

    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay on image
    original = (image[0] * 255).astype(np.uint8)
    overlaid = cv2.addWeighted(original, 0.7, heatmap, 0.3, 0)

    return overlaid
