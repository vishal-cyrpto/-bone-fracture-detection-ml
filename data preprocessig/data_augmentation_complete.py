"""
Data augmentation pipeline for bone fracture detection.
Generates additional training data through various image transformations.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import cv2
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
import random
from collections import Counter

from config import *
from image_utils import ImageProcessor

# Set up logging
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataAugmentor:
    """Data augmentation pipeline for medical images."""
    
    def __init__(self, input_dir=None, output_dir=None):
        """Initialize the data augmentor."""
        self.input_dir = Path(input_dir) if input_dir else PROCESSED_DATA_PATH / "preprocessed"
        self.output_dir = Path(output_dir) if output_dir else AUGMENTED_DATA_PATH
        
        self.processor = ImageProcessor()
        self.stats = {
            'original_counts': {},
            'augmented_counts': {},
            'augmentation_applied': Counter(),
            'total_generated': 0,
            'processing_errors': []
        }
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed for reproducibility
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
    
    def analyze_class_imbalance(self):
        """Analyze class distribution and determine augmentation needs."""
        logger.info("Analyzing class distribution...")
        
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        
        # Only augment training data
        train_dir = self.input_dir / 'train'
        if not train_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        
        class_counts = {}
        total_images = 0
        
        for class_dir in train_dir.iterdir():
            if not class_dir.is_dir() or class_dir.name not in CLASS_NAMES:
                continue
            
            # Count images in this class
            image_files = [f for f in class_dir.iterdir() 
                          if f.suffix.lower() in SUPPORTED_FORMATS]
            class_counts[class_dir.name] = len(image_files)
            total_images += len(image_files)
            
            logger.info(f"  {class_dir.name}: {len(image_files)} images")
        
        self.stats['original_counts'] = class_counts
        
        # Calculate imbalance ratio and augmentation needs
        if len(class_counts) > 1:
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            logger.info(f"Class imbalance ratio: {imbalance_ratio:.2f}:1")
            
            # Determine augmentation strategy
            augmentation_plan = {}
            for class_name, count in class_counts.items():
                if count < max_count:
                    # Calculate how many additional images needed
                    target_count = max_count
                    additional_needed = target_count - count
                    augmentation_factor = additional_needed // count + 1
                    
                    augmentation_plan[class_name] = {
                        'original_count': count,
                        'target_count': target_count,
                        'additional_needed': additional_needed,
                        'augmentation_factor': augmentation_factor
                    }
                else:
                    augmentation_plan[class_name] = {
                        'original_count': count,
                        'target_count': count,
                        'additional_needed': 0,
                        'augmentation_factor': 1
                    }
            
            logger.info("Augmentation plan:")
            for class_name, plan in augmentation_plan.items():
                logger.info(f"  {class_name}: {plan['additional_needed']} additional images needed")
            
            return augmentation_plan
        
        return {}
    
    def apply_single_augmentation(self, image, augmentation_type):
        """Apply a single augmentation to an image."""
        try:
            if augmentation_type == 'rotation':
                angle = np.random.uniform(-ROTATION_RANGE, ROTATION_RANGE)
                return self.processor.rotate_image(image, angle), f"rotate_{angle:.1f}"
            
            elif augmentation_type == 'brightness':
                factor = 1.0 + np.random.uniform(-BRIGHTNESS_RANGE, BRIGHTNESS_RANGE)
                return self.processor.enhance_brightness(image, factor), f"bright_{factor:.2f}"
            
            elif augmentation_type == 'contrast':
                factor = 1.0 + np.random.uniform(-CONTRAST_RANGE, CONTRAST_RANGE)
                return self.processor.enhance_contrast(image, factor), f"contrast_{factor:.2f}"
            
            elif augmentation_type == 'horizontal_flip' and HORIZONTAL_FLIP:
                return self.processor.flip_image(image, 'horizontal'), "hflip"
            
            elif augmentation_type == 'vertical_flip' and VERTICAL_FLIP:
                return self.processor.flip_image(image, 'vertical'), "vflip"
            
            elif augmentation_type == 'zoom':
                # Implement zoom by resizing and cropping
                zoom_factor = 1.0 + np.random.uniform(-ZOOM_RANGE, ZOOM_RANGE)
                current_size = image.shape[:2][::-1]  # (width, height)
                new_size = tuple(int(dim * zoom_factor) for dim in current_size)
                
                # Resize image
                resized = self.processor.resize_image(image, new_size)
                if resized is None:
                    return None, None
                
                # Crop or pad to original size
                if zoom_factor > 1.0:  # Zoom in - crop center
                    h, w = resized.shape[:2]
                    target_h, target_w = current_size[1], current_size[0]
                    start_h = (h - target_h) // 2
                    start_w = (w - target_w) // 2
                    cropped = resized[start_h:start_h+target_h, start_w:start_w+target_w]
                    return cropped, f"zoom_{zoom_factor:.2f}"
                else:  # Zoom out - pad
                    padded = self.processor.resize_image(resized, current_size)
                    return padded, f"zoom_{zoom_factor:.2f}"
            
            elif augmentation_type == 'translation':
                # Implement translation using affine transformation
                h, w = image.shape[:2]
                tx = int(np.random.uniform(-TRANSLATION_RANGE, TRANSLATION_RANGE) * w)
                ty = int(np.random.uniform(-TRANSLATION_RANGE, TRANSLATION_RANGE) * h)
                
                # Create translation matrix
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                
                # Apply translation
                if len(image.shape) == 3:
                    translated = cv2.warpAffine((image * 255).astype(np.uint8), M, (w, h))
                    translated = translated.astype(np.float32) / 255.0
                else:
                    translated = cv2.warpAffine((image * 255).astype(np.uint8), M, (w, h))
                    translated = translated.astype(np.float32) / 255.0
                
                return translated, f"translate_{tx}_{ty}"
            
            elif augmentation_type == 'noise' and ADD_NOISE:
                return self.processor.add_gaussian_noise(image, NOISE_VARIANCE), f"noise_{NOISE_VARIANCE}"
            
            else:
                return None, None
                
        except Exception as e:
            logger.warning(f"Error applying {augmentation_type} augmentation: {e}")
            return None, None
    
    def generate_augmented_image(self, original_image, num_augmentations=1):
        """Generate augmented versions of an image."""
        augmented_images = []
        
        # Define available augmentations with their probabilities
        available_augmentations = []
        
        if ROTATION_RANGE > 0:
            available_augmentations.extend(['rotation'] * int(ROTATION_PROBABILITY * 10))
        if BRIGHTNESS_RANGE > 0:
            available_augmentations.extend(['brightness'] * int(BRIGHTNESS_PROBABILITY * 10))
        if CONTRAST_RANGE > 0:
            available_augmentations.extend(['contrast'] * int(CONTRAST_PROBABILITY * 10))
        if HORIZONTAL_FLIP:
            available_augmentations.extend(['horizontal_flip'] * 5)
        if VERTICAL_FLIP:
            available_augmentations.extend(['vertical_flip'] * 5)
        if ZOOM_RANGE > 0:
            available_augmentations.extend(['zoom'] * int(ZOOM_PROBABILITY * 10))
        if TRANSLATION_RANGE > 0:
            available_augmentations.extend(['translation'] * int(TRANSLATION_PROBABILITY * 10))
        if ADD_NOISE:
            available_augmentations.extend(['noise'] * int(NOISE_PROBABILITY * 10))
        
        if not available_augmentations:
            logger.warning("No augmentations enabled!")
            return []
        
        for i in range(num_augmentations):
            # Start with original image
            augmented = original_image.copy()
            applied_augmentations = []
            
            # Apply random number of augmentations (1-3)
            num_transforms = np.random.randint(1, min(4, len(available_augmentations) + 1))
            selected_augmentations = np.random.choice(
                available_augmentations, 
                size=num_transforms, 
                replace=False
            )
            
            # Apply each selected augmentation
            for aug_type in selected_augmentations:
                result_image, aug_description = self.apply_single_augmentation(augmented, aug_type)
                
                if result_image is not None:
                    augmented = result_image
                    applied_augmentations.append(aug_description)
                    self.stats['augmentation_applied'][aug_type] += 1
            
            if applied_augmentations:
                augmented_images.append({
                    'image': augmented,
                    'augmentations': applied_augmentations,
                    'description': '_'.join(applied_augmentations)
                })
        
        return augmented_images
    
    def augment_class(self, class_name, augmentation_plan):
        """Augment images for a specific class."""
        logger.info(f"Augmenting class: {class_name}")
        
        # Source and destination directories
        source_dir = self.input_dir / 'train' / class_name
        dest_dir = self.output_dir / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        if not source_dir.exists():
            logger.warning(f"Source directory not found: {source_dir}")
            return
        
        # Get all images for this class
        image_files = [f for f in source_dir.iterdir() 
                      if f.suffix.lower() in SUPPORTED_FORMATS]
        
        if not image_files:
            logger.warning(f"No images found in {source_dir}")
            return
        
        # Copy original images first
        logger.info(f"Copying {len(image_files)} original images...")
        for img_file in tqdm(image_files, desc="Copying originals"):
            try:
                # Load and save original image (ensures consistent preprocessing)
                image = self.processor.load_image(img_file)
                if image is not None:
                    # Save original with consistent naming
                    output_path = dest_dir / f"orig_{img_file.stem}.png"
                    self.processor.save_image(image, output_path)
            except Exception as e:
                logger.error(f"Error copying {img_file}: {e}")
                self.stats['processing_errors'].append(str(e))
        
        # Generate augmented images if needed
        plan = augmentation_plan.get(class_name, {})
        additional_needed = plan.get('additional_needed', 0)
        
        if additional_needed > 0:
            logger.info(f"Generating {additional_needed} additional images...")
            
            generated_count = 0
            attempts = 0
            max_attempts = additional_needed * 3  # Avoid infinite loops
            
            while generated_count < additional_needed and attempts < max_attempts:
                # Select random image to augment
                img_file = np.random.choice(image_files)
                
                try:
                    # Load image
                    original_image = self.processor.load_image(img_file)
                    if original_image is None:
                        attempts += 1
                        continue
                    
                    # Generate augmented versions
                    augmented_images = self.generate_augmented_image(original_image, 1)
                    
                    for aug_data in augmented_images:
                        if generated_count >= additional_needed:
                            break
                        
                        # Save augmented image
                        aug_filename = f"aug_{generated_count:04d}_{img_file.stem}_{aug_data['description']}.png"
                        output_path = dest_dir / aug_filename
                        
                        success = self.processor.save_image(aug_data['image'], output_path)
                        if success:
                            generated_count += 1
                            self.stats['total_generated'] += 1
                    
                except Exception as e:
                    logger.error(f"Error augmenting {img_file}: {e}")
                    self.stats['processing_errors'].append(str(e))
                
                attempts += 1
            
            logger.info(f"Generated {generated_count} augmented images for {class_name}")
        
        # Update stats
        final_count = len(list(dest_dir.glob('*.png')))
        self.stats['augmented_counts'][class_name] = final_count
    
    def augment_dataset(self):
        """Augment the entire dataset based on class imbalance."""
        logger.info("Starting dataset augmentation...")
        
        # Analyze class imbalance
        augmentation_plan = self.analyze_class_imbalance()
        
        if not augmentation_plan:
            logger.warning("No augmentation plan generated. Check your data structure.")
            return
        
        # Augment each class
        for class_name in CLASS_NAMES:
            if class_name in augmentation_plan:
                self.augment_class(class_name, augmentation_plan)
        
        # Generate final report
        self.generate_augmentation_report()
        
        logger.info("Dataset augmentation completed!")
    
    def generate_augmentation_report(self):
        """Generate a comprehensive augmentation report."""
        logger.info("Generating augmentation report...")
        
        # Create reports directory
        reports_dir = self.output_dir.parent / 'reports'
        reports_dir.mkdir(exist_ok=True)
        
        # Prepare report data
        report_data = {
            'augmentation_summary': {
                'total_original': sum(self.stats['original_counts'].values()),
                'total_augmented': sum(self.stats['augmented_counts'].values()),
                'total_generated': self.stats['total_generated'],
                'processing_errors': len(self.stats['processing_errors'])
            },
            'class_breakdown': {},
            'augmentation_techniques_used': dict(self.stats['augmentation_applied']),
            'errors': self.stats['processing_errors'][:10]  # First 10 errors
        }
        
        # Class-wise breakdown
        for class_name in CLASS_NAMES:
            original_count = self.stats['original_counts'].get(class_name, 0)
            augmented_count = self.stats['augmented_counts'].get(class_name, 0)
            
            report_data['class_breakdown'][class_name] = {
                'original_count': original_count,
                'final_count': augmented_count,
                'generated_count': augmented_count - original_count,
                'augmentation_ratio': augmented_count / original_count if original_count > 0 else 0
            }
        
        # Save JSON report
        json_path = reports_dir / 'augmentation_report.json'
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Create visual report
        self.create_augmentation_visualizations(report_data, reports_dir)
        
        # Print summary
        print("\n" + "="*60)
        print("AUGMENTATION SUMMARY")
        print("="*60)
        print(f"Total original images: {report_data['augmentation_summary']['total_original']}")
        print(f"Total final images: {report_data['augmentation_summary']['total_augmented']}")
        print(f"Images generated: {report_data['augmentation_summary']['total_generated']}")
        print(f"Processing errors: {report_data['augmentation_summary']['processing_errors']}")
        print("\nClass Distribution:")
        for class_name, stats in report_data['class_breakdown'].items():
            print(f"  {class_name}: {stats['original_count']} → {stats['final_count']} "
                  f"(+{stats['generated_count']})")
        print("="*60)
        
        logger.info(f"Augmentation report saved to: {json_path}")
    
    def create_augmentation_visualizations(self, report_data, output_dir):
        """Create visualization plots for augmentation results."""
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            
            # 1. Class distribution comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            classes = list(report_data['class_breakdown'].keys())
            original_counts = [report_data['class_breakdown'][c]['original_count'] for c in classes]
            final_counts = [report_data['class_breakdown'][c]['final_count'] for c in classes]
            
            x = np.arange(len(classes))
            width = 0.35
            
            ax1.bar(x - width/2, original_counts, width, label='Original', alpha=0.8, color='skyblue')
            ax1.bar(x + width/2, final_counts, width, label='After Augmentation', alpha=0.8, color='orange')
            ax1.set_xlabel('Classes')
            ax1.set_ylabel('Number of Images')
            ax1.set_title('Class Distribution: Before vs After Augmentation')
            ax1.set_xticks(x)
            ax1.set_xticklabels(classes, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Augmentation techniques usage
            if report_data['augmentation_techniques_used']:
                techniques = list(report_data['augmentation_techniques_used'].keys())
                usage_counts = list(report_data['augmentation_techniques_used'].values())
                
                ax2.pie(usage_counts, labels=techniques, autopct='%1.1f%%', startangle=90)
                ax2.set_title('Augmentation Techniques Usage')
            else:
                ax2.text(0.5, 0.5, 'No augmentation\ntechniques used', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Augmentation Techniques Usage')
            
            plt.tight_layout()
            viz_path = output_dir / 'augmentation_visualization.png'
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Augmentation visualizations saved to: {viz_path}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def validate_augmentation_quality(self, sample_size=10):
        """Validate the quality of augmented images."""
        logger.info("Validating augmentation quality...")
        
        validation_results = {
            'samples_checked': 0,
            'quality_issues': [],
            'average_similarity': 0.0
        }
        
        try:
            for class_name in CLASS_NAMES:
                class_dir = self.output_dir / class_name
                if not class_dir.exists():
                    continue
                
                # Get original and augmented images
                original_files = list(class_dir.glob('orig_*.png'))
                augmented_files = list(class_dir.glob('aug_*.png'))
                
                if not original_files or not augmented_files:
                    continue
                
                # Sample random files for quality check
                sample_originals = np.random.choice(original_files, 
                                                  min(sample_size, len(original_files)), 
                                                  replace=False)
                sample_augmented = np.random.choice(augmented_files, 
                                                  min(sample_size, len(augmented_files)), 
                                                  replace=False)
                
                for orig_file, aug_file in zip(sample_originals, sample_augmented):
                    try:
                        orig_img = self.processor.load_image(orig_file)
                        aug_img = self.processor.load_image(aug_file)
                        
                        if orig_img is not None and aug_img is not None:
                            # Calculate similarity (MSE)
                            mse = np.mean((orig_img - aug_img) ** 2)
                            validation_results['samples_checked'] += 1
                            
                            # Check for quality issues
                            if mse > 0.1:  # Threshold for significant difference
                                validation_results['quality_issues'].append({
                                    'original': str(orig_file),
                                    'augmented': str(aug_file),
                                    'mse': float(mse)
                                })
                    
                    except Exception as e:
                        logger.warning(f"Error validating {orig_file} vs {aug_file}: {e}")
            
            logger.info(f"Validation completed. Checked {validation_results['samples_checked']} samples")
            logger.info(f"Quality issues found: {len(validation_results['quality_issues'])}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            return validation_results
    
    def cleanup_temp_files(self):
        """Clean up any temporary files created during augmentation."""
        logger.info("Cleaning up temporary files...")
        
        try:
            # Remove any temp directories or files if created
            temp_patterns = ['*.tmp', '*.temp', '.DS_Store']
            
            for pattern in temp_patterns:
                for temp_file in self.output_dir.rglob(pattern):
                    try:
                        temp_file.unlink()
                        logger.debug(f"Removed temp file: {temp_file}")
                    except Exception as e:
                        logger.warning(f"Could not remove {temp_file}: {e}")
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def main():
    """Main execution function."""
    try:
        # Create augmentor instance
        augmentor = DataAugmentor()
        
        # Run augmentation pipeline
        augmentor.augment_dataset()
        
        # Validate augmentation quality
        validation_results = augmentor.validate_augmentation_quality()
        
        # Cleanup
        augmentor.cleanup_temp_files()
        
        print("\nAugmentation pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
