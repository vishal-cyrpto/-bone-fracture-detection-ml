"""
Main data preprocessing pipeline for bone fracture detection.
Handles batch preprocessing of organized dataset with quality control.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import time
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns

from config import *
from image_utils import ImageProcessor, batch_preprocess_images

# Set up logging
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Main preprocessing pipeline for the dataset."""
    
    def __init__(self, input_dir=None, output_dir=None):
        """Initialize the data preprocessor."""
        self.input_dir = Path(input_dir) if input_dir else PROCESSED_DATA_PATH
        self.output_dir = Path(output_dir) if output_dir else PROCESSED_DATA_PATH / "preprocessed"
        
        self.processor = ImageProcessor()
        self.stats = {
            'total_images': 0,
            'processed_images': 0,
            'failed_images': 0,
            'quality_issues': 0,
            'processing_times': [],
            'split_stats': {},
            'class_stats': {}
        }
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_input_structure(self):
        """Validate the input directory structure."""
        logger.info("Validating input directory structure...")
        
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        
        # Expected structure: input_dir/split/class/images
        expected_splits = ['train', 'validation']
        if (self.input_dir / 'test').exists():
            expected_splits.append('test')
        
        structure_valid = True
        missing_paths = []
        
        for split in expected_splits:
            split_path = self.input_dir / split
            if not split_path.exists():
                missing_paths.append(str(split_path))
                structure_valid = False
                continue
            
            for class_name in CLASS_NAMES:
                class_path = split_path / class_name
                if not class_path.exists():
                    missing_paths.append(str(class_path))
                    structure_valid = False
        
        if not structure_valid:
            logger.error(f"Invalid directory structure. Missing paths: {missing_paths}")
            raise FileNotFoundError(f"Missing required directories: {missing_paths}")
        
        logger.info("✅ Input directory structure is valid")
        return True
    
    def scan_dataset(self):
        """Scan dataset and collect image information."""
        logger.info("Scanning dataset...")
        
        image_info = []
        
        for split_dir in self.input_dir.iterdir():
            if not split_dir.is_dir():
                continue
                
            split_name = split_dir.name
            if split_name not in ['train', 'validation', 'test']:
                continue
            
            logger.info(f"Scanning {split_name} split...")
            split_count = 0
            
            for class_dir in split_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                    
                class_name = class_dir.name
                if class_name not in CLASS_NAMES:
                    logger.warning(f"Unknown class: {class_name}")
                    continue
                
                # Count images in this class
                class_images = []
                for image_file in class_dir.iterdir():
                    if self.is_valid_image_file(image_file):
                        image_info.append({
                            'path': str(image_file),
                            'split': split_name,
                            'class': class_name,
                            'filename': image_file.name,
                            'size_bytes': image_file.stat().st_size
                        })
                        class_images.append(image_file)
                        split_count += 1
                
                logger.info(f"  {class_name}: {len(class_images)} images")
            
            self.stats['split_stats'][split_name] = split_count
            logger.info(f"  Total in {split_name}: {split_count} images")
        
        self.stats['total_images'] = len(image_info)
        
        # Calculate class distribution
        for class_name in CLASS_NAMES:
            class_count = sum(1 for img in image_info if img['class'] == class_name)
            self.stats['class_stats'][class_name] = class_count
        
        logger.info(f"Dataset scan complete:")
        logger.info(f"  Total images: {self.stats['total_images']}")
        logger.info(f"  Class distribution: {self.stats['class_stats']}")
        logger.info(f"  Split distribution: {self.stats['split_stats']}")
        
        return image_info
    
    def preprocess_single_image(self, image_info):
        """Preprocess a single image."""
        input_path = Path(image_info['path'])
        
        # Create corresponding output path
        relative_path = input_path.relative_to(self.input_dir)
        output_path = self.output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            start_time = time.time()
            
            # Process the image
            processed_image, processing_info = self.processor.preprocess_pipeline(
                input_path, target_size=IMAGE_SIZE
            )
            
            if processed_image is not None and processing_info['success']:
                # Save processed image
                success = self.processor.save_image(processed_image, output_path)
                
                if success:
                    processing_time = time.time() - start_time
                    return {
                        'success': True,
                        'input_path': str(input_path),
                        'output_path': str(output_path),
                        'processing_time': processing_time,
                        'quality_issues': len(processing_info['quality_issues']) > 0,
                        'processing_info': processing_info
                    }
                else:
                    return {
                        'success': False,
                        'input_path': str(input_path),
                        'error': 'Failed to save processed image'
                    }
            else:
                return {
                    'success': False,
                    'input_path': str(input_path),
                    'error': processing_info.get('error', 'Processing failed'),
                    'processing_info': processing_info
                }
                
        except Exception as e:
            return {
                'success': False,
                'input_path': str(input_path),
                'error': str(e)
            }
    
    def preprocess_batch(self, image_info_list, use_multiprocessing=True):
        """Preprocess a batch of images."""
        logger.info(f"Preprocessing {len(image_info_list)} images...")
        
        results = []
        
        if use_multiprocessing and USE_MULTIPROCESSING and len(image_info_list) > 10:
            # Use multiprocessing for large batches
            num_processes = min(MAX_PROCESSES or mp.cpu_count(), mp.cpu_count())
            logger.info(f"Using {num_processes} processes for batch preprocessing")
            
            with mp.Pool(num_processes) as pool:
                results = list(tqdm(
                    pool.imap(self.preprocess_single_image, image_info_list),
                    total=len(image_info_list),
                    desc="Preprocessing images"
                ))
        else:
            # Sequential processing
            for image_info in tqdm(image_info_list, desc="Preprocessing images"):
                result = self.preprocess_single_image(image_info)
                results.append(result)
        
        # Process results
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        quality_issues = [r for r in successful_results if r.get('quality_issues', False)]
        
        # Update statistics
        self.stats['processed_images'] += len(successful_results)
        self.stats['failed_images'] += len(failed_results)
        self.stats['quality_issues'] += len(quality_issues)
        
        if successful_results:
            processing_times = [r['processing_time'] for r in successful_results if 'processing_time' in r]
            self.stats['processing_times'].extend(processing_times)
        
        # Log batch results
        success_rate = len(successful_results) / len(results) if results else 0
        avg_time = np.mean([r.get('processing_time', 0) for r in successful_results]) if successful_results else 0
        
        logger.info(f"Batch preprocessing complete:")
        logger.info(f"  Successful: {len(successful_results)}/{len(results)} ({success_rate:.2%})")
        logger.info(f"  Failed: {len(failed_results)}")
        logger.info(f"  Quality issues: {len(quality_issues)}")
        logger.info(f"  Average time per image: {avg_time:.3f}s")
        
        if failed_results:
            logger.warning("Failed images:")
            for result in failed_results[:10]:  # Show first 10 failures
                logger.warning(f"  {result['input_path']}: {result.get('error', 'Unknown error')}")
            if len(failed_results) > 10:
                logger.warning(f"  ... and {len(failed_results) - 10} more failures")
        
        return results
    
    def preprocess_all_splits(self):
        """Preprocess all splits in the dataset."""
        logger.info("Starting preprocessing of all splits...")
        
        # Validate input structure
        self.validate_input_structure()
        
        # Scan dataset
        image_info_list = self.scan_dataset()
        
        if not image_info_list:
            logger.error("No images found to preprocess!")
            return
        
        # Group by split for organized processing
        splits_to_process = {}
        for img_info in image_info_list:
            split = img_info['split']
            if split not in splits_to_process:
                splits_to_process[split] = []
            splits_to_process[split].append(img_info)
        
        # Process each split
        all_results = {}
        for split_name, split_images in splits_to_process.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"PROCESSING {split_name.upper()} SPLIT")
            logger.info(f"{'='*50}")
            
            split_results = self.preprocess_batch(split_images)
            all_results[split_name] = split_results
        
        return all_results
    
    def create_preprocessing_report(self, results):
        """Create a comprehensive preprocessing report."""
        logger.info("Creating preprocessing report...")
        
        # Compile statistics
        total_processed = sum(len([r for r in split_results if r['success']]) for split_results in results.values())
        total_failed = sum(len([r for r in split_results if not r['success']]) for split_results in results.values())
        total_quality_issues = sum(len([r for r in split_results if r.get('quality_issues', False)]) for split_results in results.values())
        
        if self.stats['processing_times']:
            avg_processing_time = np.mean(self.stats['processing_times'])
            total_processing_time = sum(self.stats['processing_times'])
        else:
            avg_processing_time = 0
            total_processing_time = 0
        
        # Create detailed report
        report_data = {
            'preprocessing_summary': {
                'total_images_found': self.stats['total_images'],
                'images_processed': total_processed,
                'images_failed': total_failed,
                'images_with_quality_issues': total_quality_issues,
                'success_rate': total_processed / self.stats['total_images'] if self.stats['total_images'] > 0 else 0,
                'avg_processing_time_seconds': avg_processing_time,
                'total_processing_time_minutes': total_processing_time / 60
            },
            'configuration': {
                'target_image_size': IMAGE_SIZE,
                'normalize_pixels': NORMALIZE_PIXELS,
                'apply_clahe': APPLY_CLAHE,
                'clahe_parameters': {
                    'clip_limit': CLAHE_CLIP_LIMIT,
                    'grid_size': CLAHE_GRID_SIZE
                },
                'resize_method': RESIZE_METHOD,
                'supported_formats': SUPPORTED_FORMATS
            },
            'split_statistics': {},
            'processing_errors': []
        }
        
        # Add split-specific statistics
        for split_name, split_results in results.items():
            successful = [r for r in split_results if r['success']]
            failed = [r for r in split_results if not r['success']]
            quality_issues = [r for r in successful if r.get('quality_issues', False)]
            
            report_data['split_statistics'][split_name] = {
                'total_images': len(split_results),
                'processed': len(successful),
                'failed': len(failed),
                'quality_issues': len(quality_issues),
                'success_rate': len(successful) / len(split_results) if split_results else 0
            }
            
            # Collect error information
            for result in failed:
                report_data['processing_errors'].append({
                    'split': split_name,
                    'image_path': result['input_path'],
                    'error': result.get('error', 'Unknown error')
                })
        
        # Save report
        report_file = get_report_file('preprocessing_report.json')
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Create human-readable summary
        summary_lines = []
        summary_lines.append("PREPROCESSING SUMMARY REPORT")
        summary_lines.append("=" * 50)
        summary_lines.append("")
        
        summary_lines.append("OVERALL STATISTICS:")
        summary_lines.append(f"  Images found: {self.stats['total_images']}")
        summary_lines.append(f"  Images processed: {total_processed}")
        summary_lines.append(f"  Images failed: {total_failed}")
        summary_lines.append(f"  Success rate: {report_data['preprocessing_summary']['success_rate']:.2%}")
        summary_lines.append(f"  Images with quality issues: {total_quality_issues}")
        summary_lines.append(f"  Average processing time: {avg_processing_time:.3f} seconds")
        summary_lines.append(f"  Total processing time: {total_processing_time/60:.1f} minutes")
        summary_lines.append("")
        
        summary_lines.append("SPLIT BREAKDOWN:")
        for split_name, split_stats in report_data['split_statistics'].items():
            summary_lines.append(f"  {split_name.upper()}:")
            summary_lines.append(f"    Processed: {split_stats['processed']}/{split_stats['total_images']} ({split_stats['success_rate']:.2%})")
            summary_lines.append(f"    Failed: {split_stats['failed']}")
            summary_lines.append(f"    Quality issues: {split_stats['quality_issues']}")
        summary_lines.append("")
        
        if report_data['processing_errors']:
            summary_lines.append("PROCESSING ERRORS (First 10):")
            for error in report_data['processing_errors'][:10]:
                summary_lines.append(f"  {error['split']}: {Path(error['image_path']).name} - {error['error']}")
            if len(report_data['processing_errors']) > 10:
                summary_lines.append(f"  ... and {len(report_data['processing_errors']) - 10} more errors")
        
        # Save text summary
        summary_file = get_report_file('preprocessing_summary.txt')
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        # Print summary
        print('\n'.join(summary_lines))
        
        logger.info(f"Preprocessing report saved to: {report_file}")
        logger.info(f"Summary report saved to: {summary_file}")
        
        return report_data
    
    def create_sample_visualizations(self, results, num_samples_per_split=5):
        """Create before/after visualization samples."""
        logger.info("Creating preprocessing visualization samples...")
        
        for split_name, split_results in results.items():
            successful_results = [r for r in split_results if r['success']]
            
            if not successful_results:
                logger.warning(f"No successful results for {split_name} split")
                continue
            
            # Select random samples
            num_samples = min(num_samples_per_split, len(successful_results))
            sample_results = np.random.choice(successful_results, num_samples, replace=False)
            
            # Create visualization
            fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))
            if num_samples == 1:
                axes = axes.reshape(1, -1)
            
            fig.suptitle(f'Preprocessing Samples - {split_name.capitalize()} Split', fontsize=16)
            
            for i, result in enumerate(sample_results):
                try:
                    # Load original and processed images
                    original = self.processor.load_image(result['input_path'])
                    processed = self.processor.load_image(result['output_path'])
                    
                    if original is not None:
                        axes[i, 0].imshow(original, cmap='gray' if len(original.shape) == 2 else None)
                        axes[i, 0].set_title(f'Original\n{Path(result["input_path"]).name}')
                        axes[i, 0].axis('off')
                    
                    if processed is not None:
                        axes[i, 1].imshow(processed, cmap='gray' if len(processed.shape) == 2 else None)
                        axes[i, 1].set_title(f'Processed\n{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}')
                        axes[i, 1].axis('off')
                    
                except Exception as e:
                    logger.warning(f"Error creating sample visualization: {e}")
                    axes[i, 0].text(0.5, 0.5, 'Error loading\noriginal image', 
                                   ha='center', va='center', transform=axes[i, 0].transAxes)
                    axes[i, 1].text(0.5, 0.5, 'Error loading\nprocessed image',
                                   ha='center', va='center', transform=axes[i, 1].transAxes)
                    axes[i, 0].axis('off')
                    axes[i, 1].axis('off')
            
            plt.tight_layout()
            
            # Save visualization
            viz_file = get_report_file(f'preprocessing_samples_{split_name}.png')
            plt.savefig(viz_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualization saved: {viz_file}")
    
    def verify_preprocessing_output(self):
        """Verify that preprocessing was successful."""
        logger.info("Verifying preprocessing output...")
        
        verification_results = {
            'splits_found': [],
            'classes_found': [],
            'total_images': 0,
            'split_counts': {},
            'class_counts': {},
            'missing_splits': [],
            'missing_classes': [],
            'file_issues': []
        }
        
        # Check if output directory exists
        if not self.output_dir.exists():
            logger.error(f"Output directory not found: {self.output_dir}")
            return verification_results
        
        # Check each split
        expected_splits = ['train', 'validation']
        if (self.input_dir / 'test').exists():
            expected_splits.append('test')
        
        for split_name in expected_splits:
            split_path = self.output_dir / split_name
            
            if not split_path.exists():
                verification_results['missing_splits'].append(split_name)
                continue
            
            verification_results['splits_found'].append(split_name)
            split_count = 0
            
            # Check each class in this split
            for class_name in CLASS_NAMES:
                class_path = split_path / class_name
                
                if not class_path.exists():
                    verification_results['missing_classes'].append(f"{split_name}/{class_name}")
                    continue
                
                if class_name not in verification_results['classes_found']:
                    verification_results['classes_found'].append(class_name)
                
                # Count images in this class
                class_images = []
                for image_file in class_path.iterdir():
                    if self.is_valid_image_file(image_file):
                        # Quick verification that image can be loaded
                        try:
                            test_img = self.processor.load_image(image_file)
                            if test_img is not None:
                                class_images.append(image_file)
                            else:
                                verification_results['file_issues'].append(str(image_file))
                        except Exception as e:
                            verification_results['file_issues'].append(f"{image_file}: {e}")
                
                class_key = f"{split_name}/{class_name}"
                verification_results['class_counts'][class_key] = len(class_images)
                split_count += len(class_images)
            
            verification_results['split_counts'][split_name] = split_count
            verification_results['total_images'] += split_count
        
        # Log verification results
        logger.info("Preprocessing verification complete:")
        logger.info(f"  Splits found: {verification_results['splits_found']}")
        logger.info(f"  Classes found: {verification_results['classes_found']}")
        logger.info(f"  Total images: {verification_results['total_images']}")
        logger.info(f"  Split counts: {verification_results['split_counts']}")
        
        if verification_results['missing_splits']:
            logger.warning(f"  Missing splits: {verification_results['missing_splits']}")
        
        if verification_results['missing_classes']:
            logger.warning(f"  Missing classes: {verification_results['missing_classes']}")
        
        if verification_results['file_issues']:
            logger.warning(f"  File issues: {len(verification_results['file_issues'])}")
        
        # Save verification report
        verification_file = get_report_file('preprocessing_verification.json')
        with open(verification_file, 'w') as f:
            json.dump(verification_results, f, indent=2)
        
        logger.info(f"Verification report saved: {verification_file}")
        
        return verification_results
    
    def is_valid_image_file(self, file_path):
        """Check if file is a valid image file."""
        return file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FORMATS

def create_preprocessing_summary():
    """Create a comprehensive preprocessing summary with statistics and visualizations."""
    logger.info("Creating comprehensive preprocessing summary...")
    
    # Load preprocessing statistics if available
    stats_files = [
        get_report_file('preprocessing_report.json'),
        get_metadata_file('dataset_statistics.json')
    ]
    
    available_stats = {}
    for stats_file in stats_files:
        if stats_file.exists():
            try:
                with open(stats_file, 'r') as f:
                    data = json.load(f)
                    available_stats[stats_file.stem] = data
            except Exception as e:
                logger.warning(f"Could not load {stats_file}: {e}")
    
    # Create summary visualization
    if available_stats:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Preprocessing Summary Dashboard', fontsize=16)
        
        # Plot 1: Processing success rates by split
        if 'preprocessing_report' in available_stats:
            report = available_stats['preprocessing_report']
            if 'split_statistics' in report:
                splits = list(report['split_statistics'].keys())
                success_rates = [report['split_statistics'][split]['success_rate'] * 100 
                               for split in splits]
                
                axes[0, 0].bar(splits, success_rates, color=['#2E8B57', '#4682B4', '#DC143C'][:len(splits)])
                axes[0, 0].set_title('Processing Success Rate by Split')
                axes[0, 0].set_ylabel('Success Rate (%)')
                axes[0, 0].set_ylim(0, 100)
                
                # Add percentage labels on bars
                for i, rate in enumerate(success_rates):
                    axes[0, 0].text(i, rate + 1, f'{rate:.1f}%', ha='center')
        
        # Plot 2: Image counts by split and class
        if 'preprocessing_report' in available_stats:
            report = available_stats['preprocessing_report']
            if 'split_statistics' in report:
                splits = list(report['split_statistics'].keys())
                processed_counts = [report['split_statistics'][split]['processed'] for split in splits]
                failed_counts = [report['split_statistics'][split]['failed'] for split in splits]
                
                x = np.arange(len(splits))
                width = 0.35
                
                axes[0, 1].bar(x - width/2, processed_counts, width, label='Processed', color='#2E8B57')
                axes[0, 1].bar(x + width/2, failed_counts, width, label='Failed', color='#DC143C')
                
                axes[0, 1].set_title('Processing Results by Split')
                axes[0, 1].set_ylabel('Number of Images')
                axes[0, 1].set_xticks(x)
                axes[0, 1].set_xticklabels(splits)
                axes[0, 1].legend()
        
        # Plot 3: Quality issues distribution
        if 'preprocessing_report' in available_stats:
            report = available_stats['preprocessing_report']
            if 'split_statistics' in report:
                splits = list(report['split_statistics'].keys())
                quality_issues = [report['split_statistics'][split]['quality_issues'] for split in splits]
                
                axes[1, 0].bar(splits, quality_issues, color='#FFA500')
                axes[1, 0].set_title('Images with Quality Issues')
                axes[1, 0].set_ylabel('Number of Images')
                
                # Add count labels on bars
                for i, count in enumerate(quality_issues):
                    if count > 0:
                        axes[1, 0].text(i, count + 0.1, str(count), ha='center')
        
        # Plot 4: Configuration summary (text)
        config_text = "PREPROCESSING CONFIGURATION:\n\n"
        if 'preprocessing_report' in available_stats:
            config = available_stats['preprocessing_report'].get('configuration', {})
            config_text += f"Target Size: {config.get('target_image_size', 'N/A')}\n"
            config_text += f"Normalize Pixels: {config.get('normalize_pixels', 'N/A')}\n"
            config_text += f"Apply CLAHE: {config.get('apply_clahe', 'N/A')}\n"
            config_text += f"Resize Method: {config.get('resize_method', 'N/A')}\n"
            
            if 'preprocessing_summary' in available_stats['preprocessing_report']:
                summary = available_stats['preprocessing_report']['preprocessing_summary']
                config_text += f"\nPROCESSING RESULTS:\n"
                config_text += f"Success Rate: {summary.get('success_rate', 0):.2%}\n"
                config_text += f"Avg Time/Image: {summary.get('avg_processing_time_seconds', 0):.3f}s\n"
                config_text += f"Total Time: {summary.get('total_processing_time_minutes', 0):.1f}min"
        
        axes[1, 1].text(0.05, 0.95, config_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save summary dashboard
        summary_file = get_report_file('preprocessing_dashboard.png')
        plt.savefig(summary_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Preprocessing dashboard saved: {summary_file}")
        
        return summary_file
    
    return None

def main():
    """Main function to run data preprocessing."""
    logger.info("Starting data preprocessing pipeline...")
    
    try:
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Step 1: Preprocess all splits
        logger.info("Step 1: Preprocessing all dataset splits...")
        results = preprocessor.preprocess_all_splits()
        
        if not results:
            logger.error("No preprocessing results obtained!")
            return
        
        # Step 2: Create preprocessing report
        logger.info("Step 2: Creating preprocessing report...")
        report_data = preprocessor.create_preprocessing_report(results)
        
        # Step 3: Create sample visualizations
        logger.info("Step 3: Creating sample visualizations...")
        preprocessor.create_sample_visualizations(results)
        
        # Step 4: Verify preprocessing output
        logger.info("Step 4: Verifying preprocessing output...")
        verification_results = preprocessor.verify_preprocessing_output()
        
        # Step 5: Create comprehensive summary
        logger.info("Step 5: Creating comprehensive summary...")
        summary_file = create_preprocessing_summary()
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"DATA PREPROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"✅ Total images processed: {preprocessor.stats['processed_images']}")
        print(f"✅ Success rate: {report_data['preprocessing_summary']['success_rate']:.2%}")
        print(f"✅ Preprocessing time: {report_data['preprocessing_summary']['total_processing_time_minutes']:.1f} minutes")
        print(f"✅ Output directory: {preprocessor.output_dir}")
        
        if preprocessor.stats['failed_images'] > 0:
            print(f"⚠️  Failed images: {preprocessor.stats['failed_images']}")
        
        if preprocessor.stats['quality_issues'] > 0:
            print(f"⚠️  Images with quality issues: {preprocessor.stats['quality_issues']}")
        
        print(f"\n📁 Key outputs:")
        print(f"   - Preprocessed images: {preprocessor.output_dir}")
        print(f"   - Reports: {REPORTS_PATH}")
        print(f"   - Metadata: {METADATA_PATH}")
        print(f"{'='*60}\n")
        
        logger.info("Data preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        raise

if __name__ == "__main__":
    main()