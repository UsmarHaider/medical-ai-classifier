"""
Helper Utility Functions
"""
import os
import json
import random
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available computing devices.

    Returns:
        Dictionary with device information
    """
    info = {
        "tensorflow_version": tf.__version__,
        "gpus_available": len(tf.config.list_physical_devices('GPU')),
        "gpu_devices": [],
        "cpu_devices": []
    }

    # Get GPU info
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        info["gpu_devices"].append(gpu.name)

    # Get CPU info
    cpus = tf.config.list_physical_devices('CPU')
    for cpu in cpus:
        info["cpu_devices"].append(cpu.name)

    return info


def create_directory_structure(base_dir: str):
    """
    Create the project directory structure.

    Args:
        base_dir: Base directory path
    """
    directories = [
        "data",
        "models",
        "saved_models",
        "logs",
        "preprocessing",
        "training",
        "evaluation",
        "utils",
        "streamlit_app",
        "configs",
        "notebooks"
    ]

    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created: {dir_path}")

    # Create __init__.py files
    for directory in directories:
        init_path = os.path.join(base_dir, directory, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, 'w') as f:
                f.write(f'"""\n{directory.title()} Module\n"""\n')


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """
    Format metrics dictionary as a readable string.

    Args:
        metrics: Dictionary of metrics
        precision: Decimal precision

    Returns:
        Formatted string
    """
    lines = []
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            lines.append(f"  {name}: {value:.{precision}f}")
        elif isinstance(value, list):
            if all(isinstance(v, (int, float)) for v in value):
                formatted = [f"{v:.{precision}f}" for v in value]
                lines.append(f"  {name}: [{', '.join(formatted)}]")
            else:
                lines.append(f"  {name}: {value}")
        else:
            lines.append(f"  {name}: {value}")

    return "\n".join(lines)


def save_json(data: Dict, filepath: str):
    """
    Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        filepath: Path to save file
    """
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    converted_data = convert(data)

    with open(filepath, 'w') as f:
        json.dump(converted_data, f, indent=2)


def load_json(filepath: str) -> Dict:
    """
    Load dictionary from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def get_model_summary_string(model: tf.keras.Model) -> str:
    """
    Get model summary as a string.

    Args:
        model: Keras model

    Returns:
        Model summary string
    """
    string_list = []
    model.summary(print_fn=lambda x: string_list.append(x))
    return '\n'.join(string_list)


def calculate_model_size(model: tf.keras.Model) -> Dict[str, float]:
    """
    Calculate model size in MB.

    Args:
        model: Keras model

    Returns:
        Dictionary with size information
    """
    # Count parameters
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable_params = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params

    # Estimate size (assuming float32 = 4 bytes)
    size_mb = (total_params * 4) / (1024 * 1024)

    return {
        "trainable_params": int(trainable_params),
        "non_trainable_params": int(non_trainable_params),
        "total_params": int(total_params),
        "estimated_size_mb": round(size_mb, 2)
    }


def print_progress(current: int, total: int, prefix: str = "", suffix: str = "", length: int = 50):
    """
    Print a progress bar.

    Args:
        current: Current progress
        total: Total items
        prefix: Prefix string
        suffix: Suffix string
        length: Progress bar length
    """
    percent = current / total
    filled_length = int(length * percent)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent:.1%} {suffix}', end='\r')
    if current == total:
        print()


def get_class_distribution(labels: np.ndarray, class_names: List[str] = None) -> Dict:
    """
    Get class distribution from labels.

    Args:
        labels: Array of labels
        class_names: List of class names

    Returns:
        Dictionary with class distribution
    """
    unique, counts = np.unique(labels, return_counts=True)

    if class_names is None:
        class_names = [f"Class {i}" for i in unique]

    distribution = {}
    for i, (label, count) in enumerate(zip(unique, counts)):
        name = class_names[int(label)] if int(label) < len(class_names) else f"Class {label}"
        distribution[name] = {
            "count": int(count),
            "percentage": float(count / len(labels) * 100)
        }

    return distribution
