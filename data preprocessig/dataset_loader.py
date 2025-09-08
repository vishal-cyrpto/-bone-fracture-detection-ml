"""
Dataset loader and management for bone fracture detection.
Handles data loading, batch generation, and dataset splitting.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
from collections import Counter, defaultdict

from config import *
from image_utils import ImageProcessor

# Set up logging
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetLoader:
    """Dataset loader for bone fracture detection."""
    
    def __init__(self, data_dir=None):
        """Initialize the dataset loader."""
        self.data_dir = Path(data_dir) if data_dir else PROCESSED_DATA_PATH
        self.processor = ImageProcessor()
        
        # Dataset information
        self.train_data = []
        self.val_data = []
        self.test_data = []
        
        # Metadata
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(CLASS_NAMES)}
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'class_distribution': Counter(),
            'split_distribution': {},
            'image_sizes': [],
            'data_quality': {
                'corrupted_images': [],
                'size_variations': [],
                'format_issues': []
            }
        }
        
        # Set random seed
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
    
    def scan_dataset_structure(self):
        """Scan and analyze the dataset structure."""
        logger.info("Scanning dataset structure...")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")
        
        dataset_info = {
            'total_images': 0,
            'splits': {},
            'class_distribution': defaultdict(lambda: defaultdict(int))
        }
        
        # Check for different data splits
        possible_splits = ['train', 'validation', 'val', 'test']
        found_splits = []
        
        for split_name in possible_splits:
            split_dir = self.data_dir / split_name
            if split_dir.exists() and split_dir.is_dir():
                found_splits.append(split_name)
                logger.info(f"Found split: {split_name}")
        
        # If no splits found, check for direct class folders
        if not found_splits:
            logger.info("No predefined splits found. Checking for direct class structure...")
            class_dirs = [d for d in self.data_dir.iterdir() 
                         if d.is_dir() and d.name in CLASS_NAMES]
            
            if class_dirs:
                logger.info(f"Found {len(class_dirs)} class directories")
                found_splits = ['unsplit']
        
        # Scan each split
        for split_name in found_splits:
            if split_name == 'unsplit':
                split_dir = self.data_dir
            else:
                split_dir = self.data_dir / split_name
            
            split_info = {'total': 0, 'classes': {}}
            
            # Scan each class in this split
            for class_dir in split_dir.iterdir():
                if not class_dir.is_dir() or class_dir.name not in CLASS_NAMES:
                    continue
                
                class_name = class_dir.name
                
                # Count images in this class
                image_files = [f for f in class_dir.iterdir() 
                              if f.suffix.lower() in SUPPORTED_FORMATS]
                
                count = len(image_files)
                split_info['classes'][class_name] = count
                split_info['total'] += count
                dataset_info['class_distribution'][split_name][class_name] = count
                
                logger.info(f"  {split_name}/{class_name}: {count} images")
            
            dataset_info['splits'][split_name] = split_info
            dataset_info['total_images'] += split_info['total']
        
        logger.info(f"Total images found: {dataset_info['total_images']}")
        return dataset_info, found_splits
    
    def load_images_from_directory(self, directory, class_name, max_images=None):
        """Load images from a specific directory."""
        image_data = []
        
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            return image_data
        
        # Get all image files
        image_files = [f for f in directory.iterdir() 
                      if f.suffix.lower() in SUPPORTED_FORMATS]
        
        # Limit number of images if specified
        if max_images and len(image_files) > max_images:
            image_files = random.sample(image_files, max_images)
        
        # Load images
        for img_file in tqdm(image_files, desc=f"Loading {class_name} images"):
            try:
                # Load image
                image = self.processor.load_image(img_file)
                
                if image is not None:
                    # Get image info
                    image_info = {
                        'path': str(img_file),
                        'filename': img_file.name,
                        'class_name': class_name,
                        'class_idx': self.class_to_idx[class_name],
                        'shape': image.shape,
                        'size': img_file.stat().st_size,
                        'image': image  # Store the actual image data
                    }
                    
                    image_data.append(image_info)
                    self.stats['class_distribution'][class_name] += 1
                    self.stats['image_sizes'].append(image.shape)
                else:
                    self.stats['data_quality']['corrupted_images'].append(str(img_file))
                    
            except Exception as e:
                logger.warning(f"Error loading {img_file}: {e}")
                self.stats['data_quality']['corrupted_images'].append(str(img_file))
        
        self.stats['total_images'] += len(image_data)
        return image_data
    
    def create_train_val_test_split(self, data, stratify=True):
        """Create train/validation/test splits from data."""
        logger.info("Creating train/validation/test splits...")
        
        if not data:
            logger.warning("No data provided for splitting")
            return [], [], []
        
        # Extract features and labels
        X = [item['path'] for item in data]
        y = [item['class_idx'] for item in data]
        
        if stratify and len(set(y)) > 1:
            # Stratified split
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y
            )
            
            # Calculate validation split from remaining data
            val_size = VAL_SPLIT / (1 - TEST_SPLIT)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size, random_state=RANDOM_SEED, stratify=y_temp
            )
        else:
            # Random split
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED
            )
            
            val_size = VAL_SPLIT / (1 - TEST_SPLIT)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size, random_state=RANDOM_SEED
            )
        
        # Create data dictionaries
        path_to_data = {item['path']: item for item in data}
        
        train_data = [path_to_data[path] for path in X_train]
        val_data = [path_to_data[path] for path in X_val]
        test_data = [path_to_data[path] for path in X_test]
        
        # Update statistics
        self.stats['split_distribution'] = {
            'train': len(train_data),
            'validation': len(val_data),
            'test': len(test_data)
        }
        
        logger.info(f"Train: {len(train_data)} images")
        logger.info(f"Validation: {len(val_data)} images")
        logger.info(f"Test: {len(test_data)} images")
        
        return train_data, val_data, test_data
    
    def load_dataset(self, create_splits=True, max_images_per_class=None):
        """Load the complete dataset."""
        logger.info("Loading dataset...")
        
        # Scan dataset structure
        dataset_info, found_splits = self.scan_dataset_structure()
        
        all_data = []
        
        if 'train' in found_splits and 'val' in found_splits:
            # Predefined splits exist
            logger.info("Using predefined data splits")
            
            # Load training data
            train_dir = self.data_dir / 'train'
            for class_name in CLASS_NAMES:
                class_dir = train_dir / class_name
                if class_dir.exists():
                    class_data = self.load_images_from_directory(
                        class_dir, class_name, max_images_per_class
                    )
                    self.train_data.extend(class_data)
            
            # Load validation data
            val_dirs = ['val', 'validation']
            for val_name in val_dirs:
                val_dir = self.data_dir / val_name
                if val_dir.exists():
                    for class_name in CLASS_NAMES:
                        class_dir = val_dir / class_name
                        if class_dir.exists():
                            class_data = self.load_images_from_directory(
                                class_dir, class_name, max_images_per_class
                            )
                            self.val_data.extend(class_data)
                    break
            
            # Load test data if exists
            test_dir = self.data_dir / 'test'
            if test_dir.exists():
                for class_name in CLASS_NAMES:
                    class_dir = test_dir / class_name
                    if class_dir.exists():
                        class_data = self.load_images_from_directory(
                            class_dir, class_name, max_images_per_class
                        )
                        self.test_data.extend(class_data)
        
        else:
            # Load all data and create splits
            logger.info("Loading all data for splitting")
            
            for class_name in CLASS_NAMES:
                if 'unsplit' in found_splits:
                    class_dir = self.data_dir / class_name
                else:
                    # Check in available splits
                    class_dir = None
                    for split_name in found_splits:
                        potential_dir = self.data_dir / split_name / class_name
                        if potential_dir.exists():
                            class_dir = potential_dir.parent / class_name
                            break
                
                if class_dir and class_dir.exists():
                    class_data = self.load_images_from_directory(
                        class_dir, class_name, max_images_per_class
                    )
                    all_data.extend(class_data)
            
            # Create splits if needed
            if create_splits and all_data:
                self.train_data, self.val_data, self.test_data = self.create_train_val_test_split(all_data)
            else:
                self.train_data = all_data
        
        # Generate dataset report
        self.generate_dataset_report()
        
        logger.info("Dataset loading completed!")
        
        return {
            'train': self.train_data,
            'validation': self.val_data,
            'test': self.test_data
        }
    
    def get_data_generator(self, data, batch_size=None, shuffle_data=True, augment=False):
        """Create a data generator for batch processing."""
        if batch_size is None:
            batch_size = BATCH_SIZE
        
        def data_generator():
            while True:
                # Shuffle data if requested
                current_data = data.copy()
                if shuffle_data:
                    random.shuffle(current_data)
                
                # Generate batches
                for i in range(0, len(current_data), batch_size):
                    batch_data = current_data[i:i + batch_size]
                    
                    # Prepare batch
                    batch_images = []
                    batch_labels = []
                    
                    for item in batch_data:
                        try:
                            # Load image if not already loaded
                            if 'image' in item:
                                image = item['image']
                            else:
                                image = self.processor.load_image(item['path'])
                            
                            if image is not None:
                                # Apply augmentation if requested
                                if augment:
                                    # Simple augmentation (can be enhanced)
                                    if random.random() > 0.5:
                                        image = np.fliplr(image)
                                    
                                    if random.random() > 0.7:
                                        # Slight rotation
                                        angle = random.uniform(-5, 5)
                                        image = self.processor.rotate_image(image, angle)
                                
                                batch_images.append(image)
                                batch_labels.append(item['class_idx'])
                        
                        except Exception as e:
                            logger.warning(f"Error processing {item.get('path', 'unknown')}: {e}")
                    
                    if batch_images:
                        # Convert to numpy arrays
                        X_batch = np.array(batch_images, dtype=np.float32)
                        y_batch = np.array(batch_labels, dtype=np.int32)
                        
                        yield X_batch, y_batch
        
        return data_generator()
    
    def get_class_weights(self, data=None):
        """Calculate class weights for imbalanced datasets."""
        if data is None:
            data = self.train_data
        
        if not data:
            return None
        
        # Count classes
        class_counts = Counter([item['class_idx'] for item in data])
        total_samples = len(data)
        num_classes = len(class_counts)
        
        # Calculate weights
        class_weights = {}
        for class_idx, count in class_counts.items():
            weight = total_samples / (num_classes * count)
            class_weights[class_idx] = weight
        
        logger.info("Class weights calculated:")
        for class_idx, weight in class_weights.items():
            class_name = self.idx_to_class[class_idx]
            logger.info(f"  {class_name}: {weight:.4f}")
        
        return class_weights
    
    def save_dataset_splits(self, output_dir=None):
        """Save dataset splits to CSV files."""
        if output_dir is None:
            output_dir = self.data_dir / 'metadata'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each split
        splits = {
            'train': self.train_data,
            'validation': self.val_data,
            'test': self.test_data
        }
        
        for split_name, split_data in splits.items():
            if split_data:
                # Prepare DataFrame
                df_data = []
                for item in split_data:
                    df_data.append({
                        'filename': item['filename'],
                        'path': item['path'],
                        'class_name': item['class_name'],
                        'class_idx': item['class_idx'],
                        'height': item['shape'][0],
                        'width': item['shape'][1],
                        'channels': item['shape'][2] if len(item['shape']) > 2 else 1,
                        'file_size': item['size']
                    })
                
                df = pd.DataFrame(df_data)
                csv_path = output_dir / f'{split_name}_labels.csv'
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved {split_name} split to: {csv_path}")
        
        # Save class mapping
        class_mapping = {
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class
        }
        
        mapping_path = output_dir / 'class_mapping.json'
        with open(mapping_path, 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
        logger.info(f"Saved class mapping to: {mapping_path}")
    
    def generate_dataset_report(self):
        """Generate comprehensive dataset report."""
        logger.info("Generating dataset report...")
        
        # Create reports directory
        reports_dir = self.data_dir / 'reports'
        reports_dir.mkdir(exist_ok=True)
        
        # Prepare report data
        report_data = {
            'dataset_summary': {
                'total_images': self.stats['total_images'],
                'num_classes': len(CLASS_NAMES),
                'class_names': CLASS_NAMES,
                'corrupted_images': len(self.stats['data_quality']['corrupted_images'])
            },
            'class_distribution': dict(self.stats['class_distribution']),
            'split_distribution': self.stats['split_distribution'],
            'data_quality': {
                'corrupted_count': len(self.stats['data_quality']['corrupted_images']),
                'corrupted_files': self.stats['data_quality']['corrupted_images'][:10],
                'size_variations': len(set(self.stats['image_sizes']))
            },
            'image_statistics': self._calculate_image_statistics()
        }
        
        # Save JSON report
        json_path = reports_dir / 'dataset_report.json'
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Create visualizations
        self.create_dataset_visualizations(report_data, reports_dir)
        
        # Print summary
        self._print_dataset_summary(report_data)
        
        logger.info(f"Dataset report saved to: {json_path}")
    
    def _calculate_image_statistics(self):
        """Calculate statistics about image dimensions and properties."""
        if not self.stats['image_sizes']:
            return {}
        
        heights = [shape[0] for shape in self.stats['image_sizes']]
        widths = [shape[1] for shape in self.stats['image_sizes']]
        
        return {
            'height': {
                'min': int(np.min(heights)),
                'max': int(np.max(heights)),
                'mean': float(np.mean(heights)),
                'std': float(np.std(heights))
            },
            'width': {
                'min': int(np.min(widths)),
                'max': int(np.max(widths)),
                'mean': float(np.mean(widths)),
                'std': float(np.std(widths))
            },
            'aspect_ratios': {
                'values': [w/h for h, w in zip(heights, widths)],
                'mean': float(np.mean([w/h for h, w in zip(heights, widths)])),
                'std': float(np.std([w/h for h, w in zip(heights, widths)]))
            }
        }
    
    def create_dataset_visualizations(self, report_data, output_dir):
        """Create visualization plots for dataset analysis."""
        try:
            plt.style.use('default')
            
            # 1. Class distribution
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Class distribution bar chart
            if report_data['class_distribution']:
                classes = list(report_data['class_distribution'].keys())
                counts = list(report_data['class_distribution'].values())
                
                bars = ax1.bar(classes, counts, color=['skyblue', 'orange', 'lightgreen', 'pink'][:len(classes)])
                ax1.set_xlabel('Classes')
                ax1.set_ylabel('Number of Images')
                ax1.set_title('Class Distribution')
                ax1.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, count in zip(bars, counts):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            str(count), ha='center', va='bottom')
            
            # Class distribution pie chart
            if report_data['class_distribution']:
                ax2.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
                ax2.set_title('Class Distribution (Percentage)')
            
            # Split distribution
            if report_data['split_distribution']:
                splits = list(report_data['split_distribution'].keys())
                split_counts = list(report_data['split_distribution'].values())
                
                ax3.bar(splits, split_counts, color=['lightcoral', 'lightsalmon', 'lightblue'])
                ax3.set_xlabel('Data Split')
                ax3.set_ylabel('Number of Images')
                ax3.set_title('Train/Validation/Test Split')
                
                # Add value labels
                for i, count in enumerate(split_counts):
                    ax3.text(i, count + max(split_counts) * 0.01, str(count), 
                            ha='center', va='bottom')
            
            # Image size distribution
            if self.stats['image_sizes']:
                heights = [shape[0] for shape in self.stats['image_sizes']]
                widths = [shape[1] for shape in self.stats['image_sizes']]
                
                ax4.scatter(widths, heights, alpha=0.6, color='green')
                ax4.set_xlabel('Width (pixels)')
                ax4.set_ylabel('Height (pixels)')
                ax4.set_title('Image Size Distribution')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            viz_path = output_dir / 'dataset_analysis.png'
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Dataset visualizations saved to: {viz_path}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def _print_dataset_summary(self, report_data):
        """Print a formatted dataset summary."""
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        print(f"Total images: {report_data['dataset_summary']['total_images']}")
        print(f"Number of classes: {report_data['dataset_summary']['num_classes']}")
        print(f"Corrupted images: {report_data['dataset_summary']['corrupted_images']}")
        
        print("\nClass Distribution:")
        for class_name, count in report_data['class_distribution'].items():
            percentage = (count / report_data['dataset_summary']['total_images']) * 100
            print(f"  {class_name}: {count} images ({percentage:.1f}%)")
        
        if report_data['split_distribution']:
            print("\nData Splits:")
            for split_name, count in report_data['split_distribution'].items():
                print(f"  {split_name}: {count} images")
        
        if 'image_statistics' in report_data and report_data['image_statistics']:
            stats = report_data['image_statistics']
            print(f"\nImage Statistics:")
            print(f"  Height: {stats['height']['min']}-{stats['height']['max']} "
                  f"(mean: {stats['height']['mean']:.0f})")
            print(f"  Width: {stats['width']['min']}-{stats['width']['max']} "
                  f"(mean: {stats['width']['mean']:.0f})")
        
        print("="*60)

def main():
    """Main execution function."""
    try:
        # Create dataset loader
        loader = DatasetLoader()
        
        # Load dataset
        dataset = loader.load_dataset()
        
        # Save dataset splits
        loader.save_dataset_splits()
        
        # Print summary information
        print(f"\nDataset loaded successfully!")
        print(f"Train: {len(dataset['train'])} images")
        print(f"Validation: {len(dataset['validation'])} images")
        print(f"Test: {len(dataset['test'])} images")
        
        # Example: Create data generator
        if dataset['train']:
            train_generator = loader.get_data_generator(
                dataset['train'], 
                batch_size=8, 
                shuffle_data=True
            )
            
            # Get one batch to test
            X_batch, y_batch = next(train_generator)
            print(f"\nSample batch shape: {X_batch.shape}")
            print(f"Sample labels shape: {y_batch.shape}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()