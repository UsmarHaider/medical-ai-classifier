# Medical AI Classifier

A comprehensive deep learning system for medical image classification across multiple conditions and imaging modalities.

## Features

- **Multi-condition Classification**: Support for 9 different medical conditions
- **Multiple Model Architectures**: Custom CNN, VGG16, ResNet50, InceptionV3, Vision Transformer (ViT)
- **Web Application**: Interactive Streamlit interface for image upload and classification
- **AI Assistant**: Google Gemini integration for result explanation and medical information
- **Comprehensive Evaluation**: Detailed metrics, visualizations, and statistical analysis

## Supported Conditions

| Condition | Modality | Classes |
|-----------|----------|---------|
| Kidney Cancer | CT Scan | Normal, Cyst, Tumor, Stone |
| Cervical Cancer | Microscopy | Normal, Abnormal |
| Alzheimer's Disease | MRI | NonDemented, VeryMildDemented, MildDemented, ModerateDemented |
| COVID-19 | X-Ray | Normal, COVID, Viral Pneumonia |
| Pneumonia | X-Ray | Normal, Pneumonia |
| Tuberculosis | X-Ray | Normal, Tuberculosis |
| Monkeypox | Skin Image | Normal, Monkeypox |
| Malaria | Microscopy | Uninfected, Parasitized |
| Bone Shadow | X-Ray | Original, Suppressed |

## Installation

```bash
# Clone the repository
git clone https://github.com/UsmarHaider/medical-ai-classifier.git
cd medical-ai-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training a Model

```bash
# Train with default settings (Custom CNN)
python main.py --train kidney_cancer

# Train with specific model and epochs
python main.py --train covid19 --model vgg16 --epochs 100

# Train with Vision Transformer
python main.py --train alzheimer --model vit --epochs 50
```

### Evaluating a Model

```bash
python main.py --evaluate kidney_cancer
```

### Running the Web Application

```bash
python main.py --app
# Or directly
streamlit run streamlit_app/app.py
```

### List Available Options

```bash
# List all datasets
python main.py --list-datasets

# List all model architectures
python main.py --list-models
```

## Project Structure

```
medical-ai-classifier/
├── configs/              # Configuration files
├── data/                 # Dataset storage and samples
├── data_processing/      # Dask-based data processing
├── evaluation/           # Metrics and visualization
├── models/               # Model architectures
│   ├── custom_cnn.py
│   ├── transfer_learning.py
│   └── vision_transformer.py
├── preprocessing/        # Image preprocessing and augmentation
├── streamlit_app/        # Web application
│   ├── app.py
│   ├── gemini_chatbot.py
│   └── model_loader.py
├── training/             # Training utilities
├── utils/                # Helper functions
├── main.py               # Main entry point
└── requirements.txt
```

## Model Architectures

1. **Custom CNN**: Lightweight convolutional neural network for quick training
2. **VGG16**: Transfer learning with ImageNet pre-trained weights
3. **ResNet50**: Deep residual network with skip connections
4. **InceptionV3**: Multi-scale feature extraction
5. **Vision Transformer (ViT)**: Transformer-based architecture for image classification

## Web Application Features

- **Image Classification**: Upload medical images for instant classification
- **Confidence Scores**: View prediction probabilities for all classes
- **AI Assistant**: Chat with Gemini AI about results and medical conditions
- **Multi-condition Support**: Switch between different disease classifiers

## Configuration

Edit `configs/config.py` to customize:
- Training parameters (batch size, learning rate, epochs)
- Data augmentation settings
- Model configurations
- Gemini API settings

## Environment Variables

Create a `.env` file based on `.env.example`:

```
GEMINI_API_KEY=your_api_key_here
```

## Requirements

- Python 3.8+
- TensorFlow 2.12+
- PyTorch 2.0+ (for Vision Transformer)
- Streamlit 1.28+
- See `requirements.txt` for full list

## Disclaimer

This tool is for **educational and research purposes only**. It is NOT a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare providers for medical decisions.

## License

MIT License
