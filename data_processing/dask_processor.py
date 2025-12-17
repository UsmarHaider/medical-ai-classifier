"""
Large-Scale Medical Image Data Processing with Dask
AI622: Data Science and Visualization - Fall 2025

This module demonstrates the use of Dask for processing large-scale
medical image datasets (2-5 GB) as required by the course project.
"""

import dask
import dask.dataframe as dd
import dask.array as da
from dask.distributed import Client, LocalCluster
from dask import delayed
import pandas as pd
import numpy as np
import os
import glob
from PIL import Image
import json
from datetime import datetime
import hashlib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class MedicalImageDaskProcessor:
    """
    Dask-based processor for large-scale medical image datasets.
    Handles data loading, quality assessment, and preprocessing at scale.
    """

    def __init__(self, data_dir, n_workers=4, memory_limit='4GB'):
        """
        Initialize the Dask processor.

        Args:
            data_dir: Root directory containing medical image datasets
            n_workers: Number of Dask workers
            memory_limit: Memory limit per worker
        """
        self.data_dir = Path(data_dir)
        self.n_workers = n_workers
        self.memory_limit = memory_limit
        self.client = None
        self.metadata_df = None

        # Dataset configurations
        self.datasets = {
            'kidney_cancer': {
                'classes': ['Normal', 'Cyst', 'Tumor', 'Stone'],
                'modality': 'CT',
                'expected_size': (224, 224)
            },
            'cervical_cancer': {
                'classes': ['Normal', 'Abnormal'],
                'modality': 'Microscopy',
                'expected_size': (224, 224)
            },
            'alzheimer': {
                'classes': ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented'],
                'modality': 'MRI',
                'expected_size': (224, 224)
            },
            'covid19': {
                'classes': ['Normal', 'COVID', 'Viral Pneumonia'],
                'modality': 'X-Ray',
                'expected_size': (224, 224)
            },
            'pneumonia': {
                'classes': ['Normal', 'Pneumonia'],
                'modality': 'X-Ray',
                'expected_size': (224, 224)
            },
            'tuberculosis': {
                'classes': ['Normal', 'Tuberculosis'],
                'modality': 'X-Ray',
                'expected_size': (224, 224)
            },
            'monkeypox': {
                'classes': ['Normal', 'Monkeypox'],
                'modality': 'Skin',
                'expected_size': (224, 224)
            },
            'malaria': {
                'classes': ['Uninfected', 'Parasitized'],
                'modality': 'Microscopy',
                'expected_size': (224, 224)
            }
        }

    def start_cluster(self):
        """Initialize Dask distributed cluster."""
        print("=" * 60)
        print("Starting Dask Distributed Cluster")
        print("=" * 60)

        cluster = LocalCluster(
            n_workers=self.n_workers,
            threads_per_worker=2,
            memory_limit=self.memory_limit
        )
        self.client = Client(cluster)

        print(f"Dashboard: {self.client.dashboard_link}")
        print(f"Workers: {self.n_workers}")
        print(f"Memory per worker: {self.memory_limit}")
        print("=" * 60)

        return self.client

    def stop_cluster(self):
        """Shutdown Dask cluster."""
        if self.client:
            self.client.close()
            print("Dask cluster stopped.")

    @delayed
    def _process_single_image(self, image_path):
        """
        Process a single image file (delayed for parallel execution).

        Returns metadata dict with quality metrics.
        """
        try:
            img = Image.open(image_path)

            # Extract metadata
            metadata = {
                'file_path': str(image_path),
                'file_name': os.path.basename(image_path),
                'file_size_bytes': os.path.getsize(image_path),
                'width': img.size[0],
                'height': img.size[1],
                'mode': img.mode,
                'format': img.format,
                'is_valid': True,
                'error_message': None
            }

            # Convert to array for quality checks
            img_array = np.array(img.convert('RGB'))

            # Quality metrics
            metadata['mean_intensity'] = float(np.mean(img_array))
            metadata['std_intensity'] = float(np.std(img_array))
            metadata['min_intensity'] = int(np.min(img_array))
            metadata['max_intensity'] = int(np.max(img_array))

            # Check for potential issues
            metadata['is_blank'] = metadata['std_intensity'] < 5
            metadata['is_saturated'] = metadata['max_intensity'] == 255 and metadata['mean_intensity'] > 250
            metadata['is_dark'] = metadata['mean_intensity'] < 10
            metadata['aspect_ratio'] = metadata['width'] / metadata['height'] if metadata['height'] > 0 else 0

            # File hash for duplicate detection
            with open(image_path, 'rb') as f:
                metadata['file_hash'] = hashlib.md5(f.read()).hexdigest()

            img.close()

        except Exception as e:
            metadata = {
                'file_path': str(image_path),
                'file_name': os.path.basename(image_path),
                'file_size_bytes': os.path.getsize(image_path) if os.path.exists(image_path) else 0,
                'width': 0,
                'height': 0,
                'mode': None,
                'format': None,
                'is_valid': False,
                'error_message': str(e),
                'mean_intensity': 0,
                'std_intensity': 0,
                'min_intensity': 0,
                'max_intensity': 0,
                'is_blank': False,
                'is_saturated': False,
                'is_dark': False,
                'aspect_ratio': 0,
                'file_hash': None
            }

        return metadata

    def scan_dataset(self, dataset_name=None, extensions=['.jpg', '.jpeg', '.png', '.bmp', '.tiff']):
        """
        Scan dataset directory and collect all image paths.

        Args:
            dataset_name: Specific dataset to scan, or None for all
            extensions: List of valid image extensions

        Returns:
            List of image file paths
        """
        print(f"\nScanning dataset directory: {self.data_dir}")

        all_images = []

        for ext in extensions:
            pattern = str(self.data_dir / '**' / f'*{ext}')
            all_images.extend(glob.glob(pattern, recursive=True))
            pattern_upper = str(self.data_dir / '**' / f'*{ext.upper()}')
            all_images.extend(glob.glob(pattern_upper, recursive=True))

        # Remove duplicates
        all_images = list(set(all_images))

        print(f"Found {len(all_images)} image files")

        return all_images

    def process_images_parallel(self, image_paths, batch_size=1000):
        """
        Process images in parallel using Dask delayed.

        Args:
            image_paths: List of image file paths
            batch_size: Number of images per batch

        Returns:
            Dask DataFrame with image metadata
        """
        print(f"\nProcessing {len(image_paths)} images in parallel...")
        print(f"Batch size: {batch_size}")

        all_metadata = []
        n_batches = (len(image_paths) + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(image_paths))
            batch_paths = image_paths[start_idx:end_idx]

            print(f"Processing batch {batch_idx + 1}/{n_batches} ({len(batch_paths)} images)...")

            # Create delayed tasks
            delayed_results = [self._process_single_image(path) for path in batch_paths]

            # Compute in parallel
            batch_metadata = dask.compute(*delayed_results)
            all_metadata.extend(batch_metadata)

        # Convert to DataFrame
        df = pd.DataFrame(all_metadata)

        # Extract dataset and class information from file paths
        df['dataset'] = df['file_path'].apply(self._extract_dataset_name)
        df['class_label'] = df['file_path'].apply(self._extract_class_label)

        self.metadata_df = df

        print(f"\nProcessing complete. Total images: {len(df)}")

        return df

    def _extract_dataset_name(self, file_path):
        """Extract dataset name from file path."""
        path_parts = Path(file_path).parts
        for part in path_parts:
            for dataset in self.datasets.keys():
                if dataset.lower() in part.lower():
                    return dataset
        return 'unknown'

    def _extract_class_label(self, file_path):
        """Extract class label from file path."""
        path_parts = Path(file_path).parts
        # Usually class is in parent folder name
        if len(path_parts) >= 2:
            return path_parts[-2]
        return 'unknown'

    def generate_quality_report(self, df=None):
        """
        Generate comprehensive data quality report.

        Args:
            df: DataFrame with image metadata (uses self.metadata_df if None)

        Returns:
            Dictionary with quality metrics
        """
        if df is None:
            df = self.metadata_df

        if df is None or len(df) == 0:
            print("No data to analyze. Run process_images_parallel first.")
            return None

        print("\n" + "=" * 60)
        print("DATA QUALITY ASSESSMENT REPORT")
        print("=" * 60)

        report = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(df),
            'total_size_gb': df['file_size_bytes'].sum() / (1024**3),
            'datasets': {},
            'quality_issues': {},
            'summary': {}
        }

        # Overall statistics
        valid_images = df[df['is_valid'] == True]
        invalid_images = df[df['is_valid'] == False]

        report['summary'] = {
            'valid_images': len(valid_images),
            'invalid_images': len(invalid_images),
            'validity_rate': len(valid_images) / len(df) * 100 if len(df) > 0 else 0,
            'blank_images': df['is_blank'].sum(),
            'saturated_images': df['is_saturated'].sum(),
            'dark_images': df['is_dark'].sum(),
            'duplicate_count': len(df) - df['file_hash'].nunique()
        }

        print(f"\n1. OVERALL STATISTICS")
        print(f"   Total Images: {report['total_images']:,}")
        print(f"   Total Size: {report['total_size_gb']:.2f} GB")
        print(f"   Valid Images: {report['summary']['valid_images']:,} ({report['summary']['validity_rate']:.1f}%)")
        print(f"   Invalid Images: {report['summary']['invalid_images']:,}")

        print(f"\n2. QUALITY ISSUES")
        print(f"   Blank Images: {report['summary']['blank_images']:,}")
        print(f"   Saturated Images: {report['summary']['saturated_images']:,}")
        print(f"   Dark Images: {report['summary']['dark_images']:,}")
        print(f"   Potential Duplicates: {report['summary']['duplicate_count']:,}")

        # Per-dataset statistics
        print(f"\n3. PER-DATASET BREAKDOWN")
        print("-" * 60)

        for dataset in df['dataset'].unique():
            if dataset == 'unknown':
                continue

            dataset_df = df[df['dataset'] == dataset]

            dataset_stats = {
                'total_images': len(dataset_df),
                'size_gb': dataset_df['file_size_bytes'].sum() / (1024**3),
                'valid_count': dataset_df['is_valid'].sum(),
                'class_distribution': dataset_df['class_label'].value_counts().to_dict(),
                'avg_width': dataset_df['width'].mean(),
                'avg_height': dataset_df['height'].mean(),
                'quality_issues': {
                    'blank': dataset_df['is_blank'].sum(),
                    'saturated': dataset_df['is_saturated'].sum(),
                    'dark': dataset_df['is_dark'].sum()
                }
            }

            report['datasets'][dataset] = dataset_stats

            print(f"\n   {dataset.upper()}")
            print(f"   Images: {dataset_stats['total_images']:,}")
            print(f"   Size: {dataset_stats['size_gb']:.2f} GB")
            print(f"   Classes: {dataset_stats['class_distribution']}")

        print("\n" + "=" * 60)

        return report

    def check_class_balance(self, df=None):
        """
        Analyze class distribution and balance across datasets.

        Returns:
            DataFrame with class balance statistics
        """
        if df is None:
            df = self.metadata_df

        print("\n" + "=" * 60)
        print("CLASS BALANCE ANALYSIS")
        print("=" * 60)

        balance_data = []

        for dataset in df['dataset'].unique():
            if dataset == 'unknown':
                continue

            dataset_df = df[df['dataset'] == dataset]
            class_counts = dataset_df['class_label'].value_counts()

            total = class_counts.sum()
            for cls, count in class_counts.items():
                balance_data.append({
                    'dataset': dataset,
                    'class': cls,
                    'count': count,
                    'percentage': count / total * 100,
                    'imbalance_ratio': count / class_counts.min() if class_counts.min() > 0 else 0
                })

        balance_df = pd.DataFrame(balance_data)

        # Print summary
        for dataset in balance_df['dataset'].unique():
            ds_df = balance_df[balance_df['dataset'] == dataset]
            print(f"\n{dataset.upper()}:")
            for _, row in ds_df.iterrows():
                print(f"  {row['class']}: {row['count']:,} ({row['percentage']:.1f}%)")

            max_ratio = ds_df['imbalance_ratio'].max()
            if max_ratio > 3:
                print(f"  WARNING: High class imbalance (ratio: {max_ratio:.1f})")

        return balance_df

    def identify_duplicates(self, df=None):
        """
        Identify duplicate images based on file hash.

        Returns:
            DataFrame with duplicate image groups
        """
        if df is None:
            df = self.metadata_df

        # Find duplicate hashes
        duplicate_hashes = df[df.duplicated(subset=['file_hash'], keep=False)]

        if len(duplicate_hashes) > 0:
            print(f"\nFound {len(duplicate_hashes)} potentially duplicate images")
            duplicates = duplicate_hashes.groupby('file_hash').agg({
                'file_path': list,
                'file_name': 'count'
            }).reset_index()
            duplicates.columns = ['hash', 'file_paths', 'count']
            return duplicates[duplicates['count'] > 1]
        else:
            print("\nNo duplicate images found")
            return pd.DataFrame()

    def save_metadata(self, output_path):
        """Save processed metadata to parquet (efficient large-scale format)."""
        if self.metadata_df is not None:
            # Save as parquet for efficient storage
            self.metadata_df.to_parquet(output_path, index=False)
            print(f"\nMetadata saved to: {output_path}")
            print(f"File size: {os.path.getsize(output_path) / (1024**2):.2f} MB")

    def load_metadata(self, input_path):
        """Load metadata from parquet file."""
        self.metadata_df = pd.read_parquet(input_path)
        print(f"Loaded metadata for {len(self.metadata_df)} images")
        return self.metadata_df

    def get_statistical_summary(self, df=None):
        """
        Generate statistical summary of image properties.

        Returns:
            Dictionary with statistical metrics
        """
        if df is None:
            df = self.metadata_df

        valid_df = df[df['is_valid'] == True]

        stats = {
            'intensity_distribution': {
                'mean': valid_df['mean_intensity'].describe().to_dict(),
                'std': valid_df['std_intensity'].describe().to_dict()
            },
            'size_distribution': {
                'width': valid_df['width'].describe().to_dict(),
                'height': valid_df['height'].describe().to_dict(),
                'file_size_kb': (valid_df['file_size_bytes'] / 1024).describe().to_dict()
            },
            'format_distribution': valid_df['format'].value_counts().to_dict(),
            'mode_distribution': valid_df['mode'].value_counts().to_dict()
        }

        return stats


def main():
    """Main function demonstrating Dask processing pipeline."""

    # Configuration
    DATA_DIR = "/Users/usmarhaider/Desktop/DataProj/data"
    OUTPUT_DIR = "/Users/usmarhaider/Desktop/DataProj/data_processing/output"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize processor
    processor = MedicalImageDaskProcessor(
        data_dir=DATA_DIR,
        n_workers=4,
        memory_limit='2GB'
    )

    try:
        # Start Dask cluster
        client = processor.start_cluster()

        # Scan for images
        image_paths = processor.scan_dataset()

        if len(image_paths) > 0:
            # Process images in parallel
            metadata_df = processor.process_images_parallel(image_paths)

            # Generate quality report
            quality_report = processor.generate_quality_report()

            # Check class balance
            balance_df = processor.check_class_balance()

            # Identify duplicates
            duplicates = processor.identify_duplicates()

            # Save results
            processor.save_metadata(f"{OUTPUT_DIR}/image_metadata.parquet")

            # Save quality report as JSON
            with open(f"{OUTPUT_DIR}/quality_report.json", 'w') as f:
                json.dump(quality_report, f, indent=2, default=str)

            print(f"\nResults saved to: {OUTPUT_DIR}")
        else:
            print("No images found. Please ensure data is in the correct directory.")

            # Create sample output for demonstration
            print("\nGenerating sample metadata for demonstration...")
            sample_data = generate_sample_metadata()
            sample_data.to_parquet(f"{OUTPUT_DIR}/image_metadata.parquet")
            print(f"Sample metadata saved to: {OUTPUT_DIR}/image_metadata.parquet")

    finally:
        # Cleanup
        processor.stop_cluster()


def generate_sample_metadata():
    """Generate sample metadata for demonstration when no actual data exists."""

    np.random.seed(42)

    datasets = {
        'kidney_cancer': (['Normal', 'Cyst', 'Tumor', 'Stone'], 12446),
        'cervical_cancer': (['Normal', 'Abnormal'], 4012),
        'alzheimer': (['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented'], 6400),
        'covid19': (['Normal', 'COVID', 'Viral Pneumonia'], 21165),
        'pneumonia': (['Normal', 'Pneumonia'], 5863),
        'tuberculosis': (['Normal', 'Tuberculosis'], 4200),
        'monkeypox': (['Normal', 'Monkeypox'], 2142),
        'malaria': (['Uninfected', 'Parasitized'], 27558)
    }

    records = []

    for dataset, (classes, count) in datasets.items():
        for i in range(count):
            cls = np.random.choice(classes)
            records.append({
                'file_path': f'/data/{dataset}/{cls}/image_{i}.jpg',
                'file_name': f'image_{i}.jpg',
                'file_size_bytes': np.random.randint(50000, 500000),
                'width': 224,
                'height': 224,
                'mode': 'RGB',
                'format': 'JPEG',
                'is_valid': np.random.random() > 0.005,
                'mean_intensity': np.random.normal(128, 30),
                'std_intensity': np.random.normal(50, 15),
                'min_intensity': np.random.randint(0, 20),
                'max_intensity': np.random.randint(235, 256),
                'is_blank': np.random.random() < 0.001,
                'is_saturated': np.random.random() < 0.002,
                'is_dark': np.random.random() < 0.001,
                'aspect_ratio': 1.0,
                'file_hash': hashlib.md5(f'{dataset}_{cls}_{i}'.encode()).hexdigest(),
                'dataset': dataset,
                'class_label': cls
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    main()
