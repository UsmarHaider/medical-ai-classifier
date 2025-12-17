"""
Vision Transformer (ViT) for Medical Image Classification
Implements Vision Transformer architecture from scratch and with pre-trained weights
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional
import numpy as np


class PatchEmbedding(layers.Layer):
    """Patch embedding layer for Vision Transformer."""

    def __init__(self, patch_size: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = layers.Conv2D(
            embed_dim, kernel_size=patch_size, strides=patch_size
        )

    def call(self, x):
        # x shape: (batch, height, width, channels)
        x = self.projection(x)  # (batch, num_patches_h, num_patches_w, embed_dim)
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, -1, self.embed_dim])  # (batch, num_patches, embed_dim)
        return x


class MultiHeadSelfAttention(layers.Layer):
    """Multi-head self-attention layer."""

    def __init__(self, embed_dim: int, num_heads: int, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads
        )

    def call(self, x):
        return self.attention(x, x)


class TransformerBlock(layers.Layer):
    """Transformer encoder block."""

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 mlp_dim: int,
                 dropout_rate: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = keras.Sequential([
            layers.Dense(mlp_dim, activation='gelu'),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim),
            layers.Dropout(dropout_rate),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        # Multi-head self-attention
        attn_output = self.attention(self.layernorm1(x))
        attn_output = self.dropout(attn_output, training=training)
        x = x + attn_output

        # MLP
        mlp_output = self.mlp(self.layernorm2(x))
        x = x + mlp_output

        return x


class VisionTransformer:
    """
    Vision Transformer (ViT) for medical image classification.
    """

    def __init__(self,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 2,
                 patch_size: int = 16,
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 12,
                 mlp_dim: int = 3072,
                 dropout_rate: float = 0.1):
        """
        Initialize the Vision Transformer.

        Args:
            input_shape: Input image shape
            num_classes: Number of output classes
            patch_size: Size of image patches
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            mlp_dim: MLP hidden dimension
            dropout_rate: Dropout rate
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.model = None

        # Calculate number of patches
        self.num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)

    def build(self) -> keras.Model:
        """Build the Vision Transformer model."""
        inputs = layers.Input(shape=self.input_shape)

        # Patch embedding
        x = PatchEmbedding(self.patch_size, self.embed_dim)(inputs)

        # Add CLS token
        batch_size = tf.shape(inputs)[0]
        cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.embed_dim),
            initializer="zeros",
            trainable=True
        )
        cls_tokens = tf.broadcast_to(cls_token, [batch_size, 1, self.embed_dim])
        x = tf.concat([cls_tokens, x], axis=1)

        # Add positional embedding
        positions = self.add_weight(
            name="positions",
            shape=(1, self.num_patches + 1, self.embed_dim),
            initializer="zeros",
            trainable=True
        )
        x = x + positions

        # Dropout
        x = layers.Dropout(self.dropout_rate)(x)

        # Transformer blocks
        for i in range(self.num_layers):
            x = TransformerBlock(
                self.embed_dim,
                self.num_heads,
                self.mlp_dim,
                self.dropout_rate,
                name=f'transformer_block_{i}'
            )(x)

        # Layer normalization
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        # Extract CLS token output
        x = x[:, 0]

        # Classification head
        x = layers.Dense(256, activation='gelu')(x)
        x = layers.Dropout(0.3)(x)

        if self.num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs, name='vision_transformer')
        return self.model

    def build_small(self) -> keras.Model:
        """Build a smaller ViT variant for faster training."""
        # ViT-Small configuration
        self.embed_dim = 384
        self.num_heads = 6
        self.num_layers = 6
        self.mlp_dim = 1536

        return self._build_functional()

    def build_tiny(self) -> keras.Model:
        """Build a tiny ViT variant for resource-constrained environments."""
        # ViT-Tiny configuration
        self.embed_dim = 192
        self.num_heads = 3
        self.num_layers = 4
        self.mlp_dim = 768

        return self._build_functional()

    def _build_functional(self) -> keras.Model:
        """Build model using functional API."""
        inputs = layers.Input(shape=self.input_shape)

        # Patch embedding using Conv2D
        x = layers.Conv2D(
            self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding='valid'
        )(inputs)

        # Reshape to (batch, num_patches, embed_dim)
        x = layers.Reshape((-1, self.embed_dim))(x)

        # Add learnable position embedding
        num_patches = (self.input_shape[0] // self.patch_size) * (self.input_shape[1] // self.patch_size)

        # Class token
        class_token = layers.Dense(self.embed_dim, use_bias=False)(
            tf.ones((1, 1, 1))
        )
        class_token = tf.tile(class_token, [tf.shape(x)[0], 1, 1])
        x = layers.Concatenate(axis=1)([class_token, x])

        # Position embedding
        position_embedding = layers.Embedding(
            input_dim=num_patches + 1,
            output_dim=self.embed_dim
        )(tf.range(num_patches + 1))
        x = x + position_embedding

        # Dropout
        x = layers.Dropout(self.dropout_rate)(x)

        # Transformer blocks
        for _ in range(self.num_layers):
            # Layer normalization 1
            x1 = layers.LayerNormalization(epsilon=1e-6)(x)

            # Multi-head attention
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.embed_dim // self.num_heads,
                dropout=self.dropout_rate
            )(x1, x1)

            # Skip connection
            x2 = layers.Add()([attention_output, x])

            # Layer normalization 2
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

            # MLP
            x3 = layers.Dense(self.mlp_dim, activation='gelu')(x3)
            x3 = layers.Dropout(self.dropout_rate)(x3)
            x3 = layers.Dense(self.embed_dim)(x3)
            x3 = layers.Dropout(self.dropout_rate)(x3)

            # Skip connection
            x = layers.Add()([x3, x2])

        # Final layer normalization
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        # Extract class token
        x = layers.Lambda(lambda v: v[:, 0])(x)

        # Classification head
        x = layers.Dense(256, activation='gelu')(x)
        x = layers.Dropout(0.3)(x)

        if self.num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs, name='vision_transformer')
        return self.model

    def compile(self,
                learning_rate: float = 0.0001,
                optimizer: str = 'adam'):
        """Compile the model."""
        if self.model is None:
            self._build_functional()

        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'adamw':
            opt = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.01)
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

    def get_model(self) -> keras.Model:
        """Get the compiled model."""
        if self.model is None:
            self._build_functional()
            self.compile()
        return self.model

    def summary(self):
        """Print model summary."""
        if self.model is None:
            self._build_functional()
        self.model.summary()


class HuggingFaceViT:
    """
    Vision Transformer using HuggingFace Transformers library.
    Provides pre-trained ViT models.
    """

    def __init__(self,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 2,
                 model_name: str = "google/vit-base-patch16-224"):
        """
        Initialize HuggingFace ViT.

        Args:
            input_shape: Input image shape
            num_classes: Number of output classes
            model_name: HuggingFace model name
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = None

    def build(self) -> keras.Model:
        """Build model using HuggingFace ViT."""
        try:
            from transformers import TFViTModel, ViTConfig

            # Load pre-trained ViT
            config = ViTConfig.from_pretrained(self.model_name)
            vit = TFViTModel.from_pretrained(self.model_name)

            # Build custom model
            inputs = layers.Input(shape=self.input_shape)

            # ViT expects pixel_values in specific format
            # Resize and normalize
            x = layers.Rescaling(1./255)(inputs)

            # Get ViT outputs
            vit_outputs = vit(pixel_values=x)
            x = vit_outputs.last_hidden_state[:, 0]  # CLS token

            # Classification head
            x = layers.Dense(256, activation='gelu')(x)
            x = layers.Dropout(0.3)(x)

            if self.num_classes == 2:
                outputs = layers.Dense(1, activation='sigmoid')(x)
            else:
                outputs = layers.Dense(self.num_classes, activation='softmax')(x)

            self.model = keras.Model(inputs=inputs, outputs=outputs)
            return self.model

        except ImportError:
            print("HuggingFace Transformers not installed. Using custom ViT instead.")
            vit = VisionTransformer(
                input_shape=self.input_shape,
                num_classes=self.num_classes
            )
            self.model = vit.get_model()
            return self.model


# Factory functions
def create_vit(input_shape: Tuple[int, int, int] = (224, 224, 3),
               num_classes: int = 2,
               variant: str = "small") -> keras.Model:
    """
    Create Vision Transformer model.

    Args:
        input_shape: Input image shape
        num_classes: Number of classes
        variant: Model variant ('tiny', 'small', 'base')

    Returns:
        Compiled ViT model
    """
    vit = VisionTransformer(input_shape=input_shape, num_classes=num_classes)

    if variant == "tiny":
        vit.build_tiny()
    elif variant == "small":
        vit.build_small()
    else:
        vit._build_functional()

    vit.compile()
    return vit.model


def create_pretrained_vit(input_shape: Tuple[int, int, int] = (224, 224, 3),
                          num_classes: int = 2) -> keras.Model:
    """Create ViT with pre-trained weights from HuggingFace."""
    vit = HuggingFaceViT(input_shape=input_shape, num_classes=num_classes)
    return vit.build()
