"""
Medical Image Classification System
Main Entry Point

This system provides:
1. Data preprocessing for multiple medical imaging modalities
2. Model training with various architectures (CNN, VGG16, ResNet50, InceptionV3, ViT)
3. Model evaluation with comprehensive metrics
4. Streamlit web application for predictions
5. Google Gemini AI chatbot integration
"""
import os
import sys
import argparse

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from configs.config import DATASETS, TRAINING_CONFIG


def train_model(dataset: str, model_type: str, epochs: int):
    """Train a model on specified dataset."""
    from training.trainer import Trainer

    print(f"\nTraining {model_type} on {dataset} for {epochs} epochs...")

    trainer = Trainer(dataset_name=dataset, model_type=model_type)
    history = trainer.train(epochs=epochs)

    if history:
        evaluation = trainer.evaluate()
        print(f"\nEvaluation Results:")
        for metric, value in evaluation.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")

        trainer.save_model()
        print("\nModel saved successfully!")

    return trainer


def evaluate_model(dataset: str):
    """Evaluate a trained model."""
    from training.trainer import Trainer
    from evaluation.metrics import Evaluator

    print(f"\nEvaluating model for {dataset}...")

    trainer = Trainer(dataset_name=dataset)
    trainer.load_model()

    # Get test data
    data = trainer.data_loader.load_numpy_data()
    X_test, y_test = data["X_test"], data["y_test"]

    # Evaluate
    evaluator = Evaluator(trainer.model, trainer.data_loader.classes)
    metrics = evaluator.evaluate(X_test, y_test)

    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")

    print("\nClassification Report:")
    print(evaluator.get_classification_report())

    return metrics


def run_streamlit():
    """Run the Streamlit application."""
    import subprocess

    app_path = os.path.join(PROJECT_ROOT, "streamlit_app", "app.py")
    subprocess.run(["streamlit", "run", app_path])


def show_available_datasets():
    """Display available datasets."""
    print("\nAvailable Datasets:")
    print("-" * 60)

    for key, config in DATASETS.items():
        print(f"\n{config['name']} ({key})")
        print(f"  Classes: {', '.join(config['classes'])}")
        print(f"  Number of Classes: {config['num_classes']}")
        print(f"  Image Size: {config['image_size']}")
        print(f"  Model File: {config['model_file']}")


def show_model_architectures():
    """Display available model architectures."""
    from models.model_factory import ModelFactory

    print("\nAvailable Model Architectures:")
    print("-" * 60)

    for key, info in ModelFactory.AVAILABLE_MODELS.items():
        print(f"\n{info['name']} ({key})")
        print(f"  Description: {info['description']}")
        print(f"  Variants: {', '.join(info['variants'])}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Medical Image Classification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --train kidney_cancer --model vgg16 --epochs 50
  python main.py --evaluate covid19
  python main.py --app
  python main.py --list-datasets
  python main.py --list-models
        """
    )

    parser.add_argument(
        "--train",
        type=str,
        metavar="DATASET",
        help="Train model on specified dataset"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="custom_cnn",
        choices=["custom_cnn", "vgg16", "resnet50", "inceptionv3", "vit"],
        help="Model architecture to use (default: custom_cnn)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )

    parser.add_argument(
        "--evaluate",
        type=str,
        metavar="DATASET",
        help="Evaluate trained model on specified dataset"
    )

    parser.add_argument(
        "--app",
        action="store_true",
        help="Run the Streamlit web application"
    )

    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets"
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model architectures"
    )

    args = parser.parse_args()

    # Execute based on arguments
    if args.list_datasets:
        show_available_datasets()
    elif args.list_models:
        show_model_architectures()
    elif args.train:
        if args.train not in DATASETS:
            print(f"Error: Unknown dataset '{args.train}'")
            show_available_datasets()
            sys.exit(1)
        train_model(args.train, args.model, args.epochs)
    elif args.evaluate:
        if args.evaluate not in DATASETS:
            print(f"Error: Unknown dataset '{args.evaluate}'")
            show_available_datasets()
            sys.exit(1)
        evaluate_model(args.evaluate)
    elif args.app:
        run_streamlit()
    else:
        parser.print_help()
        print("\n" + "="*60)
        print("Medical Image Classification System")
        print("="*60)
        print("\nRun 'python main.py --help' for available options")
        print("\nQuick Start:")
        print("  1. Train a model: python main.py --train kidney_cancer --model vgg16")
        print("  2. Evaluate model: python main.py --evaluate kidney_cancer")
        print("  3. Run web app: python main.py --app")


if __name__ == "__main__":
    main()
