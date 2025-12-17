"""
Transfer Learning Models for Medical Image Classification
Includes VGG16, ResNet50, and InceptionV3 with pre-trained weights
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from typing import Tuple, Optional


class BaseTransferModel:
    """Base class for transfer learning models."""

    def __init__(self,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 2,
                 freeze_base: bool = True,
                 fine_tune_layers: int = 0):
        """
        Initialize the transfer learning model.

        Args:
            input_shape: Input image shape
            num_classes: Number of output classes
            freeze_base: Whether to freeze base model weights
            fine_tune_layers: Number of layers to fine-tune (from top)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.freeze_base = freeze_base
        self.fine_tune_layers = fine_tune_layers
        self.model = None
        self.base_model = None

    def _add_classification_head(self, x: tf.Tensor) -> tf.Tensor:
        """Add classification head to the model."""
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(256, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)

        if self.num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        return outputs

    def _freeze_layers(self):
        """Freeze base model layers."""
        if self.freeze_base:
            for layer in self.base_model.layers:
                layer.trainable = False

        if self.fine_tune_layers > 0:
            for layer in self.base_model.layers[-self.fine_tune_layers:]:
                layer.trainable = True

    def compile(self,
                learning_rate: float = 0.0001,
                optimizer: str = 'adam'):
        """Compile the model."""
        if self.model is None:
            self.build()

        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            opt = keras.optimizers.Adam(learning_rate=learning_rate)

        if self.num_classes == 2:
            self.model.compile(
                optimizer=opt,
                loss='binary_crossentropy',
                metrics=['accuracy', keras.metrics.AUC(name='auc')]
            )
        else:
            self.model.compile(
                optimizer=opt,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

        return self.model

    def unfreeze_for_fine_tuning(self, layers_to_unfreeze: int = 20):
        """Unfreeze top layers for fine-tuning."""
        for layer in self.base_model.layers[-layers_to_unfreeze:]:
            layer.trainable = True

        # Recompile with lower learning rate
        self.compile(learning_rate=0.00001)

    def get_model(self) -> keras.Model:
        """Get the compiled model."""
        if self.model is None:
            self.build()
            self.compile()
        return self.model

    def summary(self):
        """Print model summary."""
        if self.model is None:
            self.build()
        self.model.summary()


class VGG16Model(BaseTransferModel):
    """VGG16 transfer learning model."""

    def build(self) -> keras.Model:
        """Build VGG16-based model."""
        # Load pre-trained VGG16
        self.base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )

        # Freeze layers
        self._freeze_layers()

        # Build model
        inputs = layers.Input(shape=self.input_shape)

        # Preprocessing layer for VGG16
        x = keras.applications.vgg16.preprocess_input(inputs)

        # Base model
        x = self.base_model(x, training=False)

        # Classification head
        outputs = self._add_classification_head(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs, name='vgg16_medical')
        return self.model


class ResNet50Model(BaseTransferModel):
    """ResNet50 transfer learning model."""

    def build(self) -> keras.Model:
        """Build ResNet50-based model."""
        # Load pre-trained ResNet50
        self.base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )

        # Freeze layers
        self._freeze_layers()

        # Build model
        inputs = layers.Input(shape=self.input_shape)

        # Preprocessing layer for ResNet50
        x = keras.applications.resnet50.preprocess_input(inputs)

        # Base model
        x = self.base_model(x, training=False)

        # Classification head
        outputs = self._add_classification_head(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs, name='resnet50_medical')
        return self.model


class InceptionV3Model(BaseTransferModel):
    """InceptionV3 transfer learning model."""

    def __init__(self,
                 input_shape: Tuple[int, int, int] = (299, 299, 3),
                 num_classes: int = 2,
                 freeze_base: bool = True,
                 fine_tune_layers: int = 0):
        """
        InceptionV3 requires 299x299 input by default.
        """
        # InceptionV3 works best with 299x299 images
        if input_shape[0] < 139 or input_shape[1] < 139:
            input_shape = (299, 299, 3)

        super().__init__(input_shape, num_classes, freeze_base, fine_tune_layers)

    def build(self) -> keras.Model:
        """Build InceptionV3-based model."""
        # Load pre-trained InceptionV3
        self.base_model = InceptionV3(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )

        # Freeze layers
        self._freeze_layers()

        # Build model
        inputs = layers.Input(shape=self.input_shape)

        # Preprocessing layer for InceptionV3
        x = keras.applications.inception_v3.preprocess_input(inputs)

        # Base model
        x = self.base_model(x, training=False)

        # Classification head
        outputs = self._add_classification_head(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs, name='inceptionv3_medical')
        return self.model


class EfficientNetModel(BaseTransferModel):
    """EfficientNetB0 transfer learning model."""

    def build(self) -> keras.Model:
        """Build EfficientNetB0-based model."""
        from tensorflow.keras.applications import EfficientNetB0

        # Load pre-trained EfficientNetB0
        self.base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )

        # Freeze layers
        self._freeze_layers()

        # Build model
        inputs = layers.Input(shape=self.input_shape)

        # Base model (EfficientNet has built-in preprocessing)
        x = self.base_model(inputs, training=False)

        # Classification head
        outputs = self._add_classification_head(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs, name='efficientnet_medical')
        return self.model


class DenseNet121Model(BaseTransferModel):
    """DenseNet121 transfer learning model."""

    def build(self) -> keras.Model:
        """Build DenseNet121-based model."""
        from tensorflow.keras.applications import DenseNet121

        # Load pre-trained DenseNet121
        self.base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )

        # Freeze layers
        self._freeze_layers()

        # Build model
        inputs = layers.Input(shape=self.input_shape)

        # Preprocessing layer for DenseNet
        x = keras.applications.densenet.preprocess_input(inputs)

        # Base model
        x = self.base_model(x, training=False)

        # Classification head
        outputs = self._add_classification_head(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs, name='densenet121_medical')
        return self.model


# Factory functions
def create_vgg16(input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 2,
                 freeze_base: bool = True) -> keras.Model:
    """Create compiled VGG16 model."""
    model = VGG16Model(input_shape=input_shape, num_classes=num_classes, freeze_base=freeze_base)
    return model.get_model()


def create_resnet50(input_shape: Tuple[int, int, int] = (224, 224, 3),
                    num_classes: int = 2,
                    freeze_base: bool = True) -> keras.Model:
    """Create compiled ResNet50 model."""
    model = ResNet50Model(input_shape=input_shape, num_classes=num_classes, freeze_base=freeze_base)
    return model.get_model()


def create_inceptionv3(input_shape: Tuple[int, int, int] = (299, 299, 3),
                       num_classes: int = 2,
                       freeze_base: bool = True) -> keras.Model:
    """Create compiled InceptionV3 model."""
    model = InceptionV3Model(input_shape=input_shape, num_classes=num_classes, freeze_base=freeze_base)
    return model.get_model()


def create_efficientnet(input_shape: Tuple[int, int, int] = (224, 224, 3),
                        num_classes: int = 2,
                        freeze_base: bool = True) -> keras.Model:
    """Create compiled EfficientNet model."""
    model = EfficientNetModel(input_shape=input_shape, num_classes=num_classes, freeze_base=freeze_base)
    return model.get_model()


def create_densenet(input_shape: Tuple[int, int, int] = (224, 224, 3),
                    num_classes: int = 2,
                    freeze_base: bool = True) -> keras.Model:
    """Create compiled DenseNet121 model."""
    model = DenseNet121Model(input_shape=input_shape, num_classes=num_classes, freeze_base=freeze_base)
    return model.get_model()
