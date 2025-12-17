"""
Custom CNN Architecture for Medical Image Classification
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple, Optional


class CustomCNN:
    """
    Custom Convolutional Neural Network for medical image classification.
    Provides multiple architecture variants optimized for different tasks.
    """

    def __init__(self,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 2,
                 architecture: str = "standard"):
        """
        Initialize the Custom CNN.

        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of output classes
            architecture: Architecture variant ('standard', 'deep', 'lightweight')
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.architecture = architecture
        self.model = None

    def _conv_block(self,
                    x: tf.Tensor,
                    filters: int,
                    kernel_size: int = 3,
                    strides: int = 1,
                    use_batch_norm: bool = True,
                    use_dropout: bool = False,
                    dropout_rate: float = 0.25) -> tf.Tensor:
        """
        Create a convolutional block.

        Args:
            x: Input tensor
            filters: Number of filters
            kernel_size: Convolution kernel size
            strides: Convolution strides
            use_batch_norm: Whether to use batch normalization
            use_dropout: Whether to use dropout
            dropout_rate: Dropout rate

        Returns:
            Output tensor
        """
        x = layers.Conv2D(
            filters, kernel_size,
            strides=strides,
            padding='same',
            kernel_initializer='he_normal'
        )(x)

        if use_batch_norm:
            x = layers.BatchNormalization()(x)

        x = layers.Activation('relu')(x)

        if use_dropout:
            x = layers.Dropout(dropout_rate)(x)

        return x

    def _residual_block(self,
                        x: tf.Tensor,
                        filters: int) -> tf.Tensor:
        """
        Create a residual block.

        Args:
            x: Input tensor
            filters: Number of filters

        Returns:
            Output tensor
        """
        shortcut = x

        # First convolution
        x = self._conv_block(x, filters, use_batch_norm=True)

        # Second convolution
        x = layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)

        # Adjust shortcut if needed
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        # Add shortcut
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)

        return x

    def build_standard(self) -> keras.Model:
        """Build standard CNN architecture."""
        inputs = layers.Input(shape=self.input_shape)

        # Block 1
        x = self._conv_block(inputs, 32)
        x = self._conv_block(x, 32)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.25)(x)

        # Block 2
        x = self._conv_block(x, 64)
        x = self._conv_block(x, 64)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.25)(x)

        # Block 3
        x = self._conv_block(x, 128)
        x = self._conv_block(x, 128)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.25)(x)

        # Block 4
        x = self._conv_block(x, 256)
        x = self._conv_block(x, 256)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(0.25)(x)

        # Global Average Pooling
        x = layers.GlobalAveragePooling2D()(x)

        # Fully Connected Layers
        x = layers.Dense(512, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(256, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)

        # Output Layer
        if self.num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='custom_cnn_standard')
        return model

    def build_deep(self) -> keras.Model:
        """Build deep CNN architecture with residual connections."""
        inputs = layers.Input(shape=self.input_shape)

        # Initial convolution
        x = self._conv_block(inputs, 64, kernel_size=7, strides=2)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        # Residual blocks
        # Stage 1
        x = self._residual_block(x, 64)
        x = self._residual_block(x, 64)

        # Stage 2
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = self._residual_block(x, 128)
        x = self._residual_block(x, 128)

        # Stage 3
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = self._residual_block(x, 256)
        x = self._residual_block(x, 256)
        x = self._residual_block(x, 256)

        # Stage 4
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = self._residual_block(x, 512)
        x = self._residual_block(x, 512)
        x = self._residual_block(x, 512)

        # Global Average Pooling
        x = layers.GlobalAveragePooling2D()(x)

        # Fully Connected
        x = layers.Dense(1024, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)

        # Output
        if self.num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='custom_cnn_deep')
        return model

    def build_lightweight(self) -> keras.Model:
        """Build lightweight CNN for fast inference."""
        inputs = layers.Input(shape=self.input_shape)

        # Depthwise separable convolutions for efficiency
        x = layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        # Block 1
        x = layers.SeparableConv2D(64, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        # Block 2
        x = layers.SeparableConv2D(128, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        # Block 3
        x = layers.SeparableConv2D(256, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        # Global Average Pooling
        x = layers.GlobalAveragePooling2D()(x)

        # Fully Connected
        x = layers.Dense(256)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)

        # Output
        if self.num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='custom_cnn_lightweight')
        return model

    def build(self) -> keras.Model:
        """Build the model based on architecture type."""
        if self.architecture == "standard":
            self.model = self.build_standard()
        elif self.architecture == "deep":
            self.model = self.build_deep()
        elif self.architecture == "lightweight":
            self.model = self.build_lightweight()
        else:
            self.model = self.build_standard()

        return self.model

    def compile(self,
                learning_rate: float = 0.0001,
                optimizer: str = 'adam'):
        """
        Compile the model.

        Args:
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer type ('adam', 'sgd', 'rmsprop')
        """
        if self.model is None:
            self.build()

        # Select optimizer
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = keras.optimizers.Adam(learning_rate=learning_rate)

        # Compile based on number of classes
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


def create_custom_cnn(input_shape: Tuple[int, int, int] = (224, 224, 3),
                      num_classes: int = 2,
                      architecture: str = "standard") -> keras.Model:
    """Factory function to create and compile Custom CNN."""
    cnn = CustomCNN(input_shape=input_shape, num_classes=num_classes, architecture=architecture)
    return cnn.get_model()
