"""
Main Training Script for All Medical Image Datasets
Trains models on all 8 medical conditions
"""
import os
import sys
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import DATASETS, TRAINING_CONFIG
from training.trainer import Trainer, train_all_models


def train_kidney_cancer(model_type: str = "custom_cnn", epochs: int = 50):
    """Train model for Kidney Cancer classification."""
    print("\n" + "="*60)
    print("TRAINING: Kidney Cancer Classification")
    print("="*60)

    trainer = Trainer(
        dataset_name="kidney_cancer",
        model_type=model_type
    )

    history = trainer.train(epochs=epochs)

    if history:
        evaluation = trainer.evaluate()
        print(f"\nEvaluation Results: {evaluation}")

        trainer.save_model()
        trainer.save_training_history()

    return trainer


def train_cervical_cancer(model_type: str = "custom_cnn", epochs: int = 50):
    """Train model for Cervical Cancer classification."""
    print("\n" + "="*60)
    print("TRAINING: Cervical Cancer Classification")
    print("="*60)

    trainer = Trainer(
        dataset_name="cervical_cancer",
        model_type=model_type
    )

    history = trainer.train(epochs=epochs)

    if history:
        evaluation = trainer.evaluate()
        print(f"\nEvaluation Results: {evaluation}")

        trainer.save_model()
        trainer.save_training_history()

    return trainer


def train_alzheimer(model_type: str = "custom_cnn", epochs: int = 50):
    """Train model for Alzheimer's Disease classification."""
    print("\n" + "="*60)
    print("TRAINING: Alzheimer's Disease Classification")
    print("="*60)

    trainer = Trainer(
        dataset_name="alzheimer",
        model_type=model_type
    )

    history = trainer.train(epochs=epochs)

    if history:
        evaluation = trainer.evaluate()
        print(f"\nEvaluation Results: {evaluation}")

        trainer.save_model()
        trainer.save_training_history()

    return trainer


def train_covid19(model_type: str = "custom_cnn", epochs: int = 50):
    """Train model for COVID-19 classification."""
    print("\n" + "="*60)
    print("TRAINING: COVID-19 Classification")
    print("="*60)

    trainer = Trainer(
        dataset_name="covid19",
        model_type=model_type
    )

    history = trainer.train(epochs=epochs)

    if history:
        evaluation = trainer.evaluate()
        print(f"\nEvaluation Results: {evaluation}")

        trainer.save_model()
        trainer.save_training_history()

    return trainer


def train_pneumonia(model_type: str = "custom_cnn", epochs: int = 50):
    """Train model for Pneumonia classification."""
    print("\n" + "="*60)
    print("TRAINING: Pneumonia Classification")
    print("="*60)

    trainer = Trainer(
        dataset_name="pneumonia",
        model_type=model_type
    )

    history = trainer.train(epochs=epochs)

    if history:
        evaluation = trainer.evaluate()
        print(f"\nEvaluation Results: {evaluation}")

        trainer.save_model()
        trainer.save_training_history()

    return trainer


def train_tuberculosis(model_type: str = "custom_cnn", epochs: int = 50):
    """Train model for Tuberculosis classification."""
    print("\n" + "="*60)
    print("TRAINING: Tuberculosis Classification")
    print("="*60)

    trainer = Trainer(
        dataset_name="tuberculosis",
        model_type=model_type
    )

    history = trainer.train(epochs=epochs)

    if history:
        evaluation = trainer.evaluate()
        print(f"\nEvaluation Results: {evaluation}")

        trainer.save_model()
        trainer.save_training_history()

    return trainer


def train_monkeypox(model_type: str = "custom_cnn", epochs: int = 50):
    """Train model for Monkeypox classification."""
    print("\n" + "="*60)
    print("TRAINING: Monkeypox Classification")
    print("="*60)

    trainer = Trainer(
        dataset_name="monkeypox",
        model_type=model_type
    )

    history = trainer.train(epochs=epochs)

    if history:
        evaluation = trainer.evaluate()
        print(f"\nEvaluation Results: {evaluation}")

        trainer.save_model()
        trainer.save_training_history()

    return trainer


def train_malaria(model_type: str = "custom_cnn", epochs: int = 50):
    """Train model for Malaria classification."""
    print("\n" + "="*60)
    print("TRAINING: Malaria Classification")
    print("="*60)

    trainer = Trainer(
        dataset_name="malaria",
        model_type=model_type
    )

    history = trainer.train(epochs=epochs)

    if history:
        evaluation = trainer.evaluate()
        print(f"\nEvaluation Results: {evaluation}")

        trainer.save_model()
        trainer.save_training_history()

    return trainer


def train_bone_shadow(model_type: str = "custom_cnn", epochs: int = 50):
    """Train model for Bone Shadow Suppression."""
    print("\n" + "="*60)
    print("TRAINING: Bone Shadow Suppression")
    print("="*60)

    trainer = Trainer(
        dataset_name="bone_shadow",
        model_type=model_type
    )

    history = trainer.train(epochs=epochs)

    if history:
        evaluation = trainer.evaluate()
        print(f"\nEvaluation Results: {evaluation}")

        trainer.save_model()
        trainer.save_training_history()

    return trainer


def train_all_datasets(model_type: str = "custom_cnn", epochs: int = 50):
    """Train models on all datasets."""
    print("\n" + "="*60)
    print("TRAINING ALL DATASETS")
    print(f"Model Type: {model_type}")
    print(f"Epochs: {epochs}")
    print("="*60)

    datasets = list(DATASETS.keys())
    results = {}

    for dataset in datasets:
        try:
            print(f"\n\nStarting training for: {dataset}")
            trainer = Trainer(dataset_name=dataset, model_type=model_type)
            history = trainer.train(epochs=epochs)

            if history:
                evaluation = trainer.evaluate()
                results[dataset] = {
                    "status": "success",
                    "evaluation": evaluation
                }
                trainer.save_model()
            else:
                results[dataset] = {"status": "no_data"}

        except Exception as e:
            print(f"Error training {dataset}: {e}")
            results[dataset] = {"status": "error", "error": str(e)}

    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for dataset, result in results.items():
        status = result.get("status", "unknown")
        if status == "success":
            eval_metrics = result.get("evaluation", {})
            acc = eval_metrics.get("accuracy", "N/A")
            print(f"  {dataset}: SUCCESS - Accuracy: {acc}")
        else:
            print(f"  {dataset}: {status}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Medical Image Classification Models")

    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=list(DATASETS.keys()) + ["all"],
        help="Dataset to train on"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="custom_cnn",
        choices=["custom_cnn", "vgg16", "resnet50", "inceptionv3", "vit"],
        help="Model architecture to use"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Train all model architectures on the dataset"
    )

    args = parser.parse_args()

    print(f"\nStarting training at {datetime.now()}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")

    if args.dataset == "all":
        train_all_datasets(model_type=args.model, epochs=args.epochs)
    elif args.all_models:
        train_all_models(dataset_name=args.dataset, epochs=args.epochs)
    else:
        # Train specific dataset with specific model
        training_functions = {
            "kidney_cancer": train_kidney_cancer,
            "cervical_cancer": train_cervical_cancer,
            "alzheimer": train_alzheimer,
            "covid19": train_covid19,
            "pneumonia": train_pneumonia,
            "tuberculosis": train_tuberculosis,
            "monkeypox": train_monkeypox,
            "malaria": train_malaria,
            "bone_shadow": train_bone_shadow
        }

        if args.dataset in training_functions:
            training_functions[args.dataset](model_type=args.model, epochs=args.epochs)
        else:
            print(f"Unknown dataset: {args.dataset}")

    print(f"\nTraining completed at {datetime.now()}")


if __name__ == "__main__":
    main()
