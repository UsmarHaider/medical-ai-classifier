"""
Configuration file for Medical Image Classification System
"""
import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

# Dataset Configurations
DATASETS = {
    "kidney_cancer": {
        "name": "Kidney Cancer",
        "classes": ["Normal", "Cyst", "Tumor", "Stone"],
        "num_classes": 4,
        "image_size": (224, 224),
        "model_file": "kidney_trained_model.h5"
    },
    "cervical_cancer": {
        "name": "Cervical Cancer",
        "classes": ["Normal", "Abnormal"],
        "num_classes": 2,
        "image_size": (224, 224),
        "model_file": "cervical_trained_model.h5"
    },
    "alzheimer": {
        "name": "Alzheimer's Disease",
        "classes": ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"],
        "num_classes": 4,
        "image_size": (224, 224),
        "model_file": "alzheimer_trained_model.h5"
    },
    "covid19": {
        "name": "COVID-19",
        "classes": ["Normal", "COVID", "Viral Pneumonia"],
        "num_classes": 3,
        "image_size": (224, 224),
        "model_file": "covid19_trained_model.h5"
    },
    "pneumonia": {
        "name": "Pneumonia",
        "classes": ["Normal", "Pneumonia"],
        "num_classes": 2,
        "image_size": (224, 224),
        "model_file": "pneumonia_trained_model.h5"
    },
    "tuberculosis": {
        "name": "Tuberculosis",
        "classes": ["Normal", "Tuberculosis"],
        "num_classes": 2,
        "image_size": (224, 224),
        "model_file": "tuberculosis_trained_model.h5"
    },
    "monkeypox": {
        "name": "Monkeypox",
        "classes": ["Normal", "Monkeypox"],
        "num_classes": 2,
        "image_size": (224, 224),
        "model_file": "monkeypox_trained_model.h5"
    },
    "malaria": {
        "name": "Malaria",
        "classes": ["Uninfected", "Parasitized"],
        "num_classes": 2,
        "image_size": (224, 224),
        "model_file": "malaria_trained_model.h5"
    },
    "bone_shadow": {
        "name": "Bone Shadow Suppression",
        "classes": ["Original", "Suppressed"],
        "num_classes": 2,
        "image_size": (256, 256),
        "model_file": "bone_shadow_trained_model.h5"
    }
}

# Training Parameters
TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.0001,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "early_stopping_patience": 10,
    "reduce_lr_patience": 5,
    "random_seed": 42
}

# Model Configurations
MODEL_ARCHITECTURES = {
    "custom_cnn": {
        "name": "Custom CNN",
        "description": "Custom Convolutional Neural Network"
    },
    "vgg16": {
        "name": "VGG16",
        "description": "VGG16 with Transfer Learning"
    },
    "resnet50": {
        "name": "ResNet50",
        "description": "ResNet50 with Transfer Learning"
    },
    "inceptionv3": {
        "name": "InceptionV3",
        "description": "InceptionV3 with Transfer Learning"
    },
    "vit": {
        "name": "Vision Transformer",
        "description": "Vision Transformer (ViT)"
    }
}

# Data Augmentation Parameters
AUGMENTATION_CONFIG = {
    "rotation_range": 20,
    "width_shift_range": 0.2,
    "height_shift_range": 0.2,
    "shear_range": 0.2,
    "zoom_range": 0.2,
    "horizontal_flip": True,
    "vertical_flip": False,
    "fill_mode": "nearest"
}

# Gemini API Configuration
GEMINI_CONFIG = {
    "model_name": "gemini-pro",
    "temperature": 0.7,
    "max_output_tokens": 2048,
    "system_prompt": """You are a helpful medical AI assistant specialized in medical imaging analysis.
    You can help users understand medical image classification results, explain medical conditions,
    and provide general health information. Always remind users to consult healthcare professionals
    for actual medical advice and diagnosis."""
}
