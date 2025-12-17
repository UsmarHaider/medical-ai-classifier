"""
Data Loader Module for Medical Image Datasets
Handles loading and splitting data for training, validation, and testing
"""
import os
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .image_preprocessor import ImagePreprocessor
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import DATASETS, TRAINING_CONFIG


class DataLoader:
    """
    Data loader for medical image datasets.
    Handles loading, splitting, and creating data generators.
    """

    def __init__(self, dataset_name: str, data_dir: str = None):
        """
        Initialize the data loader.

        Args:
            dataset_name: Name of the dataset (e.g., 'kidney_cancer', 'covid19')
            data_dir: Path to the data directory
        """
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                           f"Available: {list(DATASETS.keys())}")

        self.dataset_name = dataset_name
        self.config = DATASETS[dataset_name]
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            dataset_name
        )

        self.classes = self.config["classes"]
        self.num_classes = self.config["num_classes"]
        self.image_size = self.config["image_size"]

        # Training configuration
        self.train_split = TRAINING_CONFIG["train_split"]
        self.val_split = TRAINING_CONFIG["val_split"]
        self.test_split = TRAINING_CONFIG["test_split"]
        self.batch_size = TRAINING_CONFIG["batch_size"]
        self.random_seed = TRAINING_CONFIG["random_seed"]

        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor(
            dataset_name=dataset_name,
            target_size=self.image_size
        )

    def get_image_paths_and_labels(self) -> Tuple[List[str], List[int]]:
        """
        Get all image paths and their corresponding labels.

        Returns:
            Tuple of (image_paths, labels)
        """
        image_paths = []
        labels = []

        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.data_dir, class_name)

            if not os.path.exists(class_dir):
                print(f"Warning: Class directory not found: {class_dir}")
                continue

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_paths.append(os.path.join(class_dir, img_name))
                    labels.append(class_idx)

        return image_paths, labels

    def split_data(self, image_paths: List[str],
                   labels: List[int]) -> Dict[str, Tuple[List[str], List[int]]]:
        """
        Split data into train, validation, and test sets (80/10/10).

        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels

        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        # First split: separate test set (10%)
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            image_paths, labels,
            test_size=self.test_split,
            random_state=self.random_seed,
            stratify=labels
        )

        # Second split: separate validation set (10% of original = 11.11% of remaining)
        val_ratio = self.val_split / (self.train_split + self.val_split)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels,
            test_size=val_ratio,
            random_state=self.random_seed,
            stratify=train_val_labels
        )

        return {
            "train": (train_paths, train_labels),
            "val": (val_paths, val_labels),
            "test": (test_paths, test_labels)
        }

    def create_dataset(self, image_paths: List[str],
                       labels: List[int],
                       shuffle: bool = True,
                       augment: bool = False) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset from image paths and labels.

        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels
            shuffle: Whether to shuffle the dataset
            augment: Whether to apply data augmentation

        Returns:
            TensorFlow Dataset
        """
        def load_and_preprocess(path, label):
            # Read image
            image = tf.io.read_file(path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = tf.image.resize(image, self.image_size)
            image = tf.cast(image, tf.float32) / 255.0
            return image, label

        def augment_image(image, label):
            # Random flip
            image = tf.image.random_flip_left_right(image)
            # Random brightness
            image = tf.image.random_brightness(image, max_delta=0.2)
            # Random contrast
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
            # Random rotation (using crop and pad)
            image = tf.image.random_crop(
                tf.image.pad_to_bounding_box(image, 10, 10,
                                             self.image_size[0] + 20,
                                             self.image_size[1] + 20),
                size=[self.image_size[0], self.image_size[1], 3]
            )
            return image, label

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(image_paths))

        # Load and preprocess
        dataset = dataset.map(load_and_preprocess,
                             num_parallel_calls=tf.data.AUTOTUNE)

        if augment:
            dataset = dataset.map(augment_image,
                                 num_parallel_calls=tf.data.AUTOTUNE)

        # Batch and prefetch
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def create_data_generators(self,
                               augmentation_config: Dict = None) -> Dict:
        """
        Create Keras ImageDataGenerators for train, validation, and test.

        Args:
            augmentation_config: Optional augmentation configuration

        Returns:
            Dictionary with data generators and metadata
        """
        from configs.config import AUGMENTATION_CONFIG

        aug_config = augmentation_config or AUGMENTATION_CONFIG

        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=aug_config.get("rotation_range", 20),
            width_shift_range=aug_config.get("width_shift_range", 0.2),
            height_shift_range=aug_config.get("height_shift_range", 0.2),
            shear_range=aug_config.get("shear_range", 0.2),
            zoom_range=aug_config.get("zoom_range", 0.2),
            horizontal_flip=aug_config.get("horizontal_flip", True),
            vertical_flip=aug_config.get("vertical_flip", False),
            fill_mode=aug_config.get("fill_mode", "nearest"),
            validation_split=self.val_split / (self.train_split + self.val_split)
        )

        # Validation and test data generator (only rescaling)
        test_datagen = ImageDataGenerator(rescale=1./255)

        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            print(f"Warning: Data directory not found: {self.data_dir}")
            return None

        # Create generators
        train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical' if self.num_classes > 2 else 'binary',
            subset='training',
            shuffle=True,
            seed=self.random_seed
        )

        val_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical' if self.num_classes > 2 else 'binary',
            subset='validation',
            shuffle=False,
            seed=self.random_seed
        )

        return {
            "train_generator": train_generator,
            "val_generator": val_generator,
            "train_datagen": train_datagen,
            "test_datagen": test_datagen,
            "classes": self.classes,
            "num_classes": self.num_classes,
            "image_size": self.image_size
        }

    def load_numpy_data(self,
                        max_samples: int = None) -> Dict[str, np.ndarray]:
        """
        Load all data into numpy arrays.

        Args:
            max_samples: Maximum number of samples to load (for testing)

        Returns:
            Dictionary with X_train, y_train, X_val, y_val, X_test, y_test
        """
        # Get image paths and labels
        image_paths, labels = self.get_image_paths_and_labels()

        if len(image_paths) == 0:
            raise ValueError(f"No images found in {self.data_dir}")

        # Limit samples if specified
        if max_samples and len(image_paths) > max_samples:
            indices = np.random.choice(len(image_paths), max_samples, replace=False)
            image_paths = [image_paths[i] for i in indices]
            labels = [labels[i] for i in indices]

        # Split data
        splits = self.split_data(image_paths, labels)

        # Load and preprocess images
        print("Loading training data...")
        X_train = self.preprocessor.preprocess_batch(
            splits["train"][0], show_progress=True
        )
        y_train = np.array(splits["train"][1])

        print("Loading validation data...")
        X_val = self.preprocessor.preprocess_batch(
            splits["val"][0], show_progress=True
        )
        y_val = np.array(splits["val"][1])

        print("Loading test data...")
        X_test = self.preprocessor.preprocess_batch(
            splits["test"][0], show_progress=True
        )
        y_test = np.array(splits["test"][1])

        # One-hot encode labels if multi-class
        if self.num_classes > 2:
            y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
            y_val = tf.keras.utils.to_categorical(y_val, self.num_classes)
            y_test = tf.keras.utils.to_categorical(y_test, self.num_classes)

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "classes": self.classes,
            "num_classes": self.num_classes
        }

    def get_class_weights(self, labels: List[int]) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced datasets.

        Args:
            labels: List of labels

        Returns:
            Dictionary mapping class index to weight
        """
        from sklearn.utils.class_weight import compute_class_weight

        unique_classes = np.unique(labels)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=labels
        )

        return {i: w for i, w in enumerate(weights)}

    def get_dataset_info(self) -> Dict:
        """Get information about the dataset."""
        image_paths, labels = self.get_image_paths_and_labels()

        class_counts = {}
        for label in labels:
            class_name = self.classes[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        return {
            "name": self.config["name"],
            "total_samples": len(image_paths),
            "num_classes": self.num_classes,
            "classes": self.classes,
            "class_distribution": class_counts,
            "image_size": self.image_size,
            "data_directory": self.data_dir
        }


def create_data_loader(dataset_name: str, data_dir: str = None) -> DataLoader:
    """Factory function to create data loader."""
    return DataLoader(dataset_name=dataset_name, data_dir=data_dir)
