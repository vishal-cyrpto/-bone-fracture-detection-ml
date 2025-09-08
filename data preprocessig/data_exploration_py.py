"""
Data exploration module for bone fracture detection dataset.
Analyzes dataset structure, distribution, and characteristics.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from collections import defaultdict, Counter
import logging
from tqdm import tqdm

from config import *

# Set up logging
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetExplorer:
    """Comprehensive dataset exploration and analysis."""
    
    def __init__(self, data_path=None):
        """Initialize the dataset explorer."""
        self.data_path = Path(data_path) if data_path else RAW_DATA_PATH
        self.stats = {}
        self.image_info = []
        self.patient_info = {}
        self.study_info = defaultdict(list)
        
    def explore_dataset_structure(self):
        """Analyze the dataset folder structure."""
        logger.info("Exploring dataset structure...")
        
        structure_info = {
            'total_patients': 0,
            'total_studies': 0,
            'total_images': 0,
            'study_types': set(),
            'label_distribution': Counter(),
            'patients_per_study_type': defaultdict(int),
            'images_per_patient': [],
            'studies_per_patient': []
        }
        
        if not self.data_path.exists():
            logger.error(f"Data path does not exist: {self.data_path}")
            return structure_info
            
        # Walk through the directory structure
        for study_type_dir in self.data_path.iterdir():
            if not study_type_dir.is_dir():
                continue
                
            study_type = study_type_dir.name
            structure_info['study_types'].add(study_type)
            
            # Skip if focusing on specific study type
            if FOCUS_STUDY_TYPE and study_type != FOCUS_STUDY_TYPE:
                continue
                
            logger.info(f"Processing study type: {study_type}")
            
            # Process each patient
            for patient_dir in tqdm(study_type_dir.iterdir(), desc=f"Patients in {study_type}"):
                if not patient_dir.is_dir():
                    continue
                    
                patient_id = patient_dir.name
                structure_info['total_patients'] += 1
                structure_info['patients_per_study_type'][study_type] += 1
                
                patient_studies = 0
                patient_images = 0
                
                # Process each study for this patient
                for study_dir in patient_dir.iterdir():
                    if not study_dir.is_dir():
                        continue
                        
                    study_name = study_dir.name
                    structure_info['total_studies'] += 1
                    patient_studies += 1
                    
                    # Extract label from study name
                    label = self.extract_label_from_study_name(study_name)
                    structure_info['label_distribution'][label] += 1
                    
                    # Count images in this study
                    study_images = 0
                    for image_file in study_dir.iterdir():
                        if self.is_valid_image_file(image_file):
                            study_images += 1
                            patient_images += 1
                            structure_info['total_images'] += 1
                    
                    self.study_info[study_type].append({
                        'patient_id': patient_id,
                        'study_name': study_name,
                        'label': label,
                        'num_images': study_images,
                        'study_path': str(study_dir)
                    })
                
                structure_info['studies_per_patient'].append(patient_studies)
                structure_info['images_per_patient'].append(patient_images)
                
                self.patient_info[patient_id] = {
                    'study_type': study_type,
                    'num_studies': patient_studies,
                    'num_images': patient_images
                }
        
        # Convert sets to lists for JSON serialization
        structure_info['study_types'] = list(structure_info['study_types'])
        
        self.stats['structure'] = structure_info
        logger.info(f"Dataset structure analysis complete:")
        logger.info(f"  - Total patients: {structure_info['total_patients']}")
        logger.info(f"  - Total studies: {structure_info['total_studies']}")
        logger.info(f"  - Total images: {structure_info['total_images']}")
        logger.info(f"  - Study types: {structure_info['study_types']}")
        logger.info(f"  - Label distribution: {dict(structure_info['label_distribution'])}")
        
        return structure_info
    
    def analyze_images(self, sample_size=None):
        """Analyze image characteristics."""
        logger.info("Analyzing image characteristics...")
        
        image_stats = {
            'dimensions': [],
            'file_sizes': [],
            'formats': Counter(),
            'brightness': [],
            'contrast': [],
            'corrupted_files': [],
            'dimension_distribution': Counter(),
            'aspect_ratios': []
        }
        
        images_analyzed = 0
        total_images = self.stats['structure']['total_images'] if 'structure' in self.stats else float('inf')
        
        if sample_size:
            total_images = min(sample_size, total_images)
            
        # Walk through all images
        for study_type_dir in self.data_path.iterdir():
            if not study_type_dir.is_dir():
                continue
                
            # Skip if focusing on specific study type
            if FOCUS_STUDY_TYPE and study_type_dir.name != FOCUS_STUDY_TYPE:
                continue
                
            for patient_dir in study_type_dir.iterdir():
                if not patient_dir.is_dir():
                    continue
                    
                for study_dir in patient_dir.iterdir():
                    if not study_dir.is_dir():
                        continue
                        
                    for image_file in study_dir.iterdir():
                        if not self.is_valid_image_file(image_file):
                            continue
                            
                        if images_analyzed >= total_images:
                            break
                            
                        try:
                            # Analyze this image
                            img_info = self.analyze_single_image(image_file)
                            
                            # Update statistics
                            image_stats['dimensions'].append(img_info['dimensions'])
                            image_stats['file_sizes'].append(img_info['file_size'])
                            image_stats['formats'][img_info['format']] += 1
                            image_stats['brightness'].append(img_info['brightness'])
                            image_stats['contrast'].append(img_info['contrast'])
                            image_stats['aspect_ratios'].append(img_info['aspect_ratio'])
                            
                            # Dimension distribution
                            dim_key = f"{img_info['dimensions'][0]}x{img_info['dimensions'][1]}"
                            image_stats['dimension_distribution'][dim_key] += 1
                            
                            self.image_info.append({
                                'file_path': str(image_file),
                                'patient_id': patient_dir.name,
                                'study_name': study_dir.name,
                                'study_type': study_type_dir.name,
                                'label': self.extract_label_from_study_name(study_dir.name),
                                **img_info
                            })
                            
                            images_analyzed += 1
                            
                        except Exception as e:
                            logger.warning(f"Error analyzing {image_file}: {e}")
                            image_stats['corrupted_files'].append(str(image_file))
                            
                        if images_analyzed % 100 == 0:
                            logger.info(f"Analyzed {images_analyzed} images...")
        
        # Calculate summary statistics
        if image_stats['dimensions']:
            widths, heights = zip(*image_stats['dimensions'])
            image_stats['summary'] = {
                'total_analyzed': images_analyzed,
                'corrupted_count': len(image_stats['corrupted_files']),
                'width_stats': {
                    'mean': np.mean(widths),
                    'std': np.std(widths),
                    'min': np.min(widths),
                    'max': np.max(widths),
                    'median': np.median(widths)
                },
                'height_stats': {
                    'mean': np.mean(heights),
                    'std': np.std(heights),
                    'min': np.min(heights),
                    'max': np.max(heights),
                    'median': np.median(heights)
                },
                'filesize_stats': {
                    'mean': np.mean(image_stats['file_sizes']),
                    'std': np.std(image_stats['file_sizes']),
                    'min': np.min(image_stats['file_sizes']),
                    'max': np.max(image_stats['file_sizes']),
                    'median': np.median(image_stats['file_sizes'])
                },
                'brightness_stats': {
                    'mean': np.mean(image_stats['brightness']),
                    'std': np.std(image_stats['brightness']),
                    'min': np.min(image_stats['brightness']),
                    'max': np.max(image_stats['brightness'])
                },
                'contrast_stats': {
                    'mean': np.mean(image_stats['contrast']),
                    'std': np.std(image_stats['contrast']),
                    'min': np.min(image_stats['contrast']),
                    'max': np.max(image_stats['contrast'])
                }
            }
        
        self.stats['images'] = image_stats
        logger.info(f"Image analysis complete: {images_analyzed} images analyzed")
        
        return image_stats
    
    def analyze_single_image(self, image_path):
        """Analyze characteristics of a single image."""
        # Get file size
        file_size = os.path.getsize(image_path)
        
        # Open and analyze image
        with Image.open(image_path) as img:
            width, height = img.size
            format_type = img.format
            
            # Convert to grayscale for analysis
            if img.mode != 'L':
                gray_img = img.convert('L')
            else:
                gray_img = img
                
            # Convert to numpy array
            img_array = np.array(gray_img)
            
            # Calculate brightness (mean pixel value)
            brightness = np.mean(img_array)
            
            # Calculate contrast (standard deviation)
            contrast = np.std(img_array)
            
            # Calculate aspect ratio
            aspect_ratio = width / height
        
        return {
            'dimensions': (width, height),
            'file_size': file_size,
            'format': format_type,
            'brightness': brightness,
            'contrast': contrast,
            'aspect_ratio': aspect_ratio
        }
    
    def generate_visualizations(self):
        """Generate visualization plots for the dataset."""
        logger.info("Generating visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Dataset Analysis Visualizations', fontsize=16)
        
        # 1. Label distribution
        if 'structure' in self.stats:
            labels = list(self.stats['structure']['label_distribution'].keys())
            counts = list(self.stats['structure']['label_distribution'].values())
            axes[0, 0].pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
            axes[0, 0].set_title('Label Distribution')
        
        # 2. Images per patient distribution
        if 'structure' in self.stats and self.stats['structure']['images_per_patient']:
            axes[0, 1].hist(self.stats['structure']['images_per_patient'], bins=20, alpha=0.7)
            axes[0, 1].set_title('Images per Patient')
            axes[0, 1].set_xlabel('Number of Images')
            axes[0, 1].set_ylabel('Number of Patients')
        
        # 3. Image dimensions scatter plot
        if 'images' in self.stats and self.stats['images']['dimensions']:
            widths, heights = zip(*self.stats['images']['dimensions'])
            axes[0, 2].scatter(widths, heights, alpha=0.5, s=10)
            axes[0, 2].set_title('Image Dimensions Distribution')
            axes[0, 2].set_xlabel('Width (pixels)')
            axes[0, 2].set_ylabel('Height (pixels)')
        
        # 4. File size distribution
        if 'images' in self.stats and self.stats['images']['file_sizes']:
            file_sizes_mb = [size / (1024 * 1024) for size in self.stats['images']['file_sizes']]
            axes[1, 0].hist(file_sizes_mb, bins=30, alpha=0.7)
            axes[1, 0].set_title('File Size Distribution')
            axes[1, 0].set_xlabel('File Size (MB)')
            axes[1, 0].set_ylabel('Number of Images')
        
        # 5. Brightness distribution
        if 'images' in self.stats and self.stats['images']['brightness']:
            axes[1, 1].hist(self.stats['images']['brightness'], bins=30, alpha=0.7, color='orange')
            axes[1, 1].set_title('Brightness Distribution')
            axes[1, 1].set_xlabel('Average Brightness')
            axes[1, 1].set_ylabel('Number of Images')
        
        # 6. Contrast distribution
        if 'images' in self.stats and self.stats['images']['contrast']:
            axes[1, 2].hist(self.stats['images']['contrast'], bins=30, alpha=0.7, color='green')
            axes[1, 2].set_title('Contrast Distribution')
            axes[1, 2].set_xlabel('Contrast (Std Dev)')
            axes[1, 2].set_ylabel('Number of Images')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = get_report_file('dataset_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualizations saved to: {plot_path}")
        
        return fig
    
    def export_statistics(self):
        """Export statistics to files."""
        logger.info("Exporting statistics...")
        
        # Export comprehensive statistics as JSON
        stats_file = get_metadata_file('dataset_statistics.json')
        with open(stats_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_stats = self.convert_numpy_types(self.stats)
            json.dump(json_stats, f, indent=2)
        
        # Export image information as CSV
        if self.image_info:
            df_images = pd.DataFrame(self.image_info)
            csv_file = get_metadata_file('image_analysis.csv')
            df_images.to_csv(csv_file, index=False)
            logger.info(f"Image analysis exported to: {csv_file}")
        
        # Export study information as CSV
        if self.study_info:
            all_studies = []
            for study_type, studies in self.study_info.items():
                for study in studies:
                    study['study_type'] = study_type
                    all_studies.append(study)
            
            df_studies = pd.DataFrame(all_studies)
            study_csv_file = get_metadata_file('study_analysis.csv')
            df_studies.to_csv(study_csv_file, index=False)
            logger.info(f"Study analysis exported to: {study_csv_file}")
        
        logger.info(f"Statistics exported to: {stats_file}")
        return stats_file
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        logger.info("Generating summary report...")
        
        report_lines = []
        report_lines.append("# Dataset Analysis Summary Report")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Structure summary
        if 'structure' in self.stats:
            struct = self.stats['structure']
            report_lines.append("## Dataset Structure")
            report_lines.append(f"Total Patients: {struct['total_patients']}")
            report_lines.append(f"Total Studies: {struct['total_studies']}")
            report_lines.append(f"Total Images: {struct['total_images']}")
            report_lines.append(f"Study Types: {', '.join(struct['study_types'])}")
            report_lines.append("")
            
            report_lines.append("### Label Distribution")
            for label, count in struct['label_distribution'].items():
                percentage = (count / struct['total_studies']) * 100
                report_lines.append(f"  {label}: {count} ({percentage:.1f}%)")
            report_lines.append("")
        
        # Image analysis summary
        if 'images' in self.stats and 'summary' in self.stats['images']:
            img_summary = self.stats['images']['summary']
            report_lines.append("## Image Analysis")
            report_lines.append(f"Images Analyzed: {img_summary['total_analyzed']}")
            report_lines.append(f"Corrupted Files: {img_summary['corrupted_count']}")
            report_lines.append("")
            
            report_lines.append("### Image Dimensions")
            width_stats = img_summary['width_stats']
            height_stats = img_summary['height_stats']
            report_lines.append(f"Width - Mean: {width_stats['mean']:.1f}, Range: {width_stats['min']}-{width_stats['max']}")
            report_lines.append(f"Height - Mean: {height_stats['mean']:.1f}, Range: {height_stats['min']}-{height_stats['max']}")
            report_lines.append("")
            
            report_lines.append("### Image Quality Metrics")
            brightness_stats = img_summary['brightness_stats']
            contrast_stats = img_summary['contrast_stats']
            report_lines.append(f"Brightness - Mean: {brightness_stats['mean']:.1f}, Std: {brightness_stats['std']:.1f}")
            report_lines.append(f"Contrast - Mean: {contrast_stats['mean']:.1f}, Std: {contrast_stats['std']:.1f}")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        if 'images' in self.stats:
            if len(self.stats['images']['corrupted_files']) > 0:
                report_lines.append("- Remove or fix corrupted image files")
            
            if 'summary' in self.stats['images']:
                img_summary = self.stats['images']['summary']
                if img_summary['width_stats']['std'] > img_summary['width_stats']['mean'] * 0.5:
                    report_lines.append("- Consider standardizing image dimensions")
                
                if img_summary['brightness_stats']['std'] > 50:
                    report_lines.append("- Apply histogram equalization for brightness normalization")
        
        # Write report
        report_file = get_report_file('dataset_summary_report.txt')
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Summary report saved to: {report_file}")
        return report_file
    
    def extract_label_from_study_name(self, study_name):
        """Extract label from study directory name."""
        study_lower = study_name.lower()
        if 'negative' in study_lower:
            return 'negative'
        elif 'positive' in study_lower:
            return 'positive'
        else:
            logger.warning(f"Could not extract label from study name: {study_name}")
            return 'unknown'
    
    def is_valid_image_file(self, file_path):
        """Check if file is a valid image file."""
        return file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FORMATS
    
    def convert_numpy_types(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Counter):
            return dict(obj)
        else:
            return obj

def main():
    """Main function to run dataset exploration."""
    logger.info("Starting dataset exploration...")
    
    # Initialize explorer
    explorer = DatasetExplorer()
    
    # Run exploration steps
    try:
        # Step 1: Analyze structure
        explorer.explore_dataset_structure()
        
        # Step 2: Analyze images (sample first 1000 for speed)
        explorer.analyze_images(sample_size=1000 if not DEBUG_MODE else 100)
        
        # Step 3: Generate visualizations
        if GENERATE_HTML_REPORTS:
            explorer.generate_visualizations()
        
        # Step 4: Export statistics
        explorer.export_statistics()
        
        # Step 5: Generate summary report
        explorer.generate_summary_report()
        
        logger.info("Dataset exploration completed successfully!")
        
        # Print quick summary
        if 'structure' in explorer.stats:
            struct = explorer.stats['structure']
            print(f"\n{'='*50}")
            print(f"DATASET EXPLORATION SUMMARY")
            print(f"{'='*50}")
            print(f"Total Images: {struct['total_images']}")
            print(f"Total Patients: {struct['total_patients']}")
            print(f"Label Distribution: {dict(struct['label_distribution'])}")
            if 'images' in explorer.stats and 'summary' in explorer.stats['images']:
                img_summary = explorer.stats['images']['summary']
                print(f"Corrupted Files: {img_summary['corrupted_count']}")
            print(f"{'='*50}\n")
        
    except Exception as e:
        logger.error(f"Error during dataset exploration: {e}")
        raise

if __name__ == "__main__":
    main()
        