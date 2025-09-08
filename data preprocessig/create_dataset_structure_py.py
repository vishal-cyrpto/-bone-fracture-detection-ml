"""
Create organized dataset structure for bone fracture detection.
Transforms nested patient/study structure into clean train/val/test splits.
"""

import os
import shutil
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
import logging
from tqdm import tqdm
import numpy as np

from config import *

# Set up logging
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetOrganizer:
    """Organize dataset from nested structure to ML-ready format."""
    
    def __init__(self, source_path=None, target_path=None):
        """Initialize the dataset organizer."""
        self.source_path = Path(source_path) if source_path else RAW_DATA_PATH
        self.target_path = Path(target_path) if target_path else PROCESSED_DATA_PATH
        
        self.image_records = []
        self.patient_records = {}
        self.study_records = []
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'total_patients': 0,
            'total_studies': 0,
            'corrupted_files': [],
            'skipped_files': [],
            'class_distribution': Counter(),
            'patient_class_distribution': Counter()
        }
    
    def scan_source_directory(self):
        """Scan source directory and collect all image information."""
        logger.info(f"Scanning source directory: {self.source_path}")
        
        if not self.source_path.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_path}")
        
        # Walk through directory structure
        for study_type_dir in self.source_path.iterdir():
            if not study_type_dir.is_dir():
                continue
            
            study_type = study_type_dir.name
            logger.info(f"Processing study type: {study_type}")
            
            # Skip if focusing on specific study type
            if FOCUS_STUDY_TYPE and study_type != FOCUS_STUDY_TYPE:
                logger.info(f"Skipping {study_type} (focusing on {FOCUS_STUDY_TYPE})")
                continue
            
            # Process each patient
            for patient_dir in tqdm(study_type_dir.iterdir(), desc=f"Patients in {study_type}"):
                if not patient_dir.is_dir():
                    continue
                
                patient_id = patient_dir.name
                if patient_id not in self.patient_records:
                    self.patient_records[patient_id] = {
                        'patient_id': patient_id,
                        'study_type': study_type,
                        'studies': [],
                        'total_images': 0,
                        'labels': []
                    }
                    self.stats['total_patients'] += 1
                
                # Process each study
                for study_dir in patient_dir.iterdir():
                    if not study_dir.is_dir():
                        continue
                    
                    study_name = study_dir.name
                    label = self.extract_label_from_study_name(study_name)
                    
                    # Create study record
                    study_record = {
                        'patient_id': patient_id,
                        'study_type': study_type,
                        'study_name': study_name,
                        'label': label,
                        'images': [],
                        'num_images': 0
                    }
                    
                    # Process images in this study
                    for image_file in study_dir.iterdir():
                        if not self.is_valid_image_file(image_file):
                            continue
                        
                        try:
                            # Validate image can be opened
                            self.validate_image_file(image_file)
                            
                            # Create image record
                            image_record = {
                                'original_path': str(image_file),
                                'patient_id': patient_id,
                                'study_type': study_type,
                                'study_name': study_name,
                                'image_name': image_file.name,
                                'label': label,
                                'file_size': os.path.getsize(image_file)
                            }
                            
                            self.image_records.append(image_record)
                            study_record['images'].append(image_record)
                            study_record['num_images'] += 1
                            self.stats['total_images'] += 1
                            self.stats['class_distribution'][label] += 1
                            
                        except Exception as e:
                            logger.warning(f"Skipping corrupted image {image_file}: {e}")
                            self.stats['corrupted_files'].append(str(image_file))
                    
                    if study_record['num_images'] > 0:
                        self.study_records.append(study_record)
                        self.patient_records[patient_id]['studies'].append(study_record)
                        self.patient_records[patient_id]['total_images'] += study_record['num_images']
                        self.patient_records[patient_id]['labels'].append(label)
                        self.stats['total_studies'] += 1
        
        # Calculate patient-level class distribution
        for patient_id, patient_info in self.patient_records.items():
            # Determine primary label for patient
            labels = patient_info['labels']
            if labels:
                primary_label = max(set(labels), key=labels.count)
                self.stats['patient_class_distribution'][primary_label] += 1
        
        logger.info(f"Scanning complete:")
        logger.info(f"  - Total images: {self.stats['total_images']}")
        logger.info(f"  - Total patients: {self.stats['total_patients']}")
        logger.info(f"  - Total studies: {self.stats['total_studies']}")
        logger.info(f"  - Class distribution: {dict(self.stats['class_distribution'])}")
        logger.info(f"  - Corrupted files: {len(self.stats['corrupted_files'])}")
    
    def create_splits(self):
        """Create train/validation/test splits."""
        logger.info("Creating train/validation/test splits...")
        
        if not self.image_records:
            raise ValueError("No image records found. Run scan_source_directory first.")
        
        # Create DataFrame for easier manipulation
        df = pd.DataFrame(self.image_records)
        
        if PATIENT_LEVEL_SPLIT:
            splits = self._create_patient_level_splits(df)
        else:
            splits = self._create_image_level_splits(df)
        
        # Log split statistics
        for split_name, split_df in splits.items():
            class_dist = split_df['label'].value_counts().to_dict()
            logger.info(f"{split_name.capitalize()} set: {len(split_df)} images, {class_dist}")
        
        return splits
    
    def _create_patient_level_splits(self, df):
        """Create splits ensuring no patient appears in multiple splits."""
        logger.info("Creating patient-level splits...")
        
        # Get unique patients and their primary labels
        patient_labels = []
        patient_ids = []
        
        for patient_id, patient_info in self.patient_records.items():
            labels = patient_info['labels']
            if labels:
                primary_label = max(set(labels), key=labels.count)
                patient_labels.append(primary_label)
                patient_ids.append(patient_id)
        
        # First split: train vs (val + test)
        train_patients, temp_patients, train_labels, temp_labels = train_test_split(
            patient_ids, patient_labels,
            test_size=(VALIDATION_RATIO + TEST_RATIO),
            random_state=RANDOM_SEED,
            stratify=patient_labels if STRATIFY_SPLITS else None
        )
        
        # Second split: val vs test
        if TEST_RATIO > 0:
            val_size = VALIDATION_RATIO / (VALIDATION_RATIO + TEST_RATIO)
            val_patients, test_patients, _, _ = train_test_split(
                temp_patients, temp_labels,
                test_size=1-val_size,
                random_state=RANDOM_SEED,
                stratify=temp_labels if STRATIFY_SPLITS else None
            )
        else:
            val_patients = temp_patients
            test_patients = []
        
        # Create splits based on patient assignments
        splits = {
            'train': df[df['patient_id'].isin(train_patients)],
            'validation': df[df['patient_id'].isin(val_patients)]
        }
        
        if test_patients:
            splits['test'] = df[df['patient_id'].isin(test_patients)]
        
        return splits
    
    def _create_image_level_splits(self, df):
        """Create splits at image level (may have patient overlap)."""
        logger.info("Creating image-level splits...")
        
        # Stratify by label if enabled
        stratify_col = df['label'] if STRATIFY_SPLITS else None
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(VALIDATION_RATIO + TEST_RATIO),
            random_state=RANDOM_SEED,
            stratify=stratify_col
        )
        
        # Second split: val vs test
        if TEST_RATIO > 0:
            val_size = VALIDATION_RATIO / (VALIDATION_RATIO + TEST_RATIO)
            val_df, test_df = train_test_split(
                temp_df,
                test_size=1-val_size,
                random_state=RANDOM_SEED,
                stratify=temp_df['label'] if STRATIFY_SPLITS else None
            )
            splits = {'train': train_df, 'validation': val_df, 'test': test_df}
        else:
            splits = {'train': train_df, 'validation': temp_df}
        
        return splits
    
    def organize_files(self, splits):
        """Copy/move files to organized directory structure."""
        logger.info(f"Organizing files to: {self.target_path}")
        
        # Create directory structure
        self.create_directory_structure()
        
        copy_stats = {
            'total_copied': 0,
            'copy_errors': [],
            'splits': {}
        }
        
        for split_name, split_df in splits.items():
            logger.info(f"Processing {split_name} split...")
            
            split_stats = {'total': len(split_df), 'copied': 0, 'errors': 0}
            
            for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Copying {split_name}"):
                try:
                    success = self.copy_image_file(row, split_name)
                    if success:
                        split_stats['copied'] += 1
                        copy_stats['total_copied'] += 1
                    else:
                        split_stats['errors'] += 1
                        
                except Exception as e:
                    logger.warning(f"Error copying {row['original_path']}: {e}")
                    copy_stats['copy_errors'].append({
                        'file': row['original_path'],
                        'error': str(e)
                    })
                    split_stats['errors'] += 1
            
            copy_stats['splits'][split_name] = split_stats
            logger.info(f"{split_name.capitalize()} - Copied: {split_stats['copied']}, Errors: {split_stats['errors']}")
        
        return copy_stats
    
    def create_directory_structure(self):
        """Create the target directory structure."""
        # Main splits
        for split in ['train', 'validation', 'test']:
            for class_name in CLASS_NAMES:
                split_path = self.target_path / split / class_name
                split_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {split_path}")
    
    def copy_image_file(self, image_record, split_name):
        """Copy an image file to the organized structure."""
        source_path = Path(image_record['original_path'])
        
        if not source_path.exists():
            logger.warning(f"Source file not found: {source_path}")
            return False
        
        # Create unique filename to avoid conflicts
        patient_id = image_record['patient_id']
        study_name = image_record['study_name']
        original_name = Path(image_record['image_name']).stem
        extension = Path(image_record['image_name']).suffix
        
        new_filename = f"{patient_id}_{study_name}_{original_name}{extension}"
        
        # Target path
        target_path = self.target_path / split_name / image_record['label'] / new_filename
        
        # Copy file
        try:
            shutil.copy2(source_path, target_path)
            return True
        except Exception as e:
            logger.error(f"Failed to copy {source_path} to {target_path}: {e}")
            return False
    
    def generate_metadata_files(self, splits):
        """Generate metadata files for each split."""
        logger.info("Generating metadata files...")
        
        metadata_files = {}
        
        for split_name, split_df in splits.items():
            # Create metadata DataFrame with additional info
            metadata_df = split_df.copy()
            
            # Add target path information
            metadata_df['target_path'] = metadata_df.apply(
                lambda row: str(self.target_path / split_name / row['label'] / 
                              f"{row['patient_id']}_{row['study_name']}_{Path(row['image_name']).stem}{Path(row['image_name']).suffix}"),
                axis=1
            )
            
            # Add encoded labels
            metadata_df['label_encoded'] = metadata_df['label'].map(LABEL_ENCODING)
            
            # Save to CSV
            csv_file = get_metadata_file(f'{split_name}_metadata.csv')
            metadata_df.to_csv(csv_file, index=False)
            metadata_files[split_name] = csv_file
            
            logger.info(f"{split_name.capitalize()} metadata saved to: {csv_file}")
        
        # Create combined metadata file
        combined_df = pd.concat(splits.values(), ignore_index=True)
        combined_df['split'] = pd.concat([
            pd.Series([split_name] * len(split_df)) 
            for split_name, split_df in splits.items()
        ], ignore_index=True)
        
        combined_file = get_metadata_file('combined_metadata.csv')
        combined_df.to_csv(combined_file, index=False)
        metadata_files['combined'] = combined_file
        
        logger.info(f"Combined metadata saved to: {combined_file}")
        
        return metadata_files
    
    def generate_class_mapping(self):
        """Generate class mapping files."""
        logger.info("Generating class mapping files...")
        
        # Class to index mapping
        class_mapping = {
            'class_names': CLASS_NAMES,
            'label_encoding': LABEL_ENCODING,
            'label_decoding': LABEL_DECODING,
            'num_classes': NUM_CLASSES
        }
        
        mapping_file = get_metadata_file('class_mapping.json')
        with open(mapping_file, 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
        logger.info(f"Class mapping saved to: {mapping_file}")
        return mapping_file
    
    def generate_statistics_report(self, splits, copy_stats):
        """Generate comprehensive statistics report."""
        logger.info("Generating statistics report...")
        
        # Compile comprehensive statistics
        final_stats = {
            'source_directory': str(self.source_path),
            'target_directory': str(self.target_path),
            'processing_timestamp': pd.Timestamp.now().isoformat(),
            'configuration': {
                'train_ratio': TRAIN_RATIO,
                'validation_ratio': VALIDATION_RATIO,
                'test_ratio': TEST_RATIO,
                'patient_level_split': PATIENT_LEVEL_SPLIT,
                'stratify_splits': STRATIFY_SPLITS,
                'random_seed': RANDOM_SEED,
                'focus_study_type': FOCUS_STUDY_TYPE
            },
            'source_statistics': dict(self.stats),
            'split_statistics': {},
            'copy_statistics': copy_stats
        }
        
        # Add split-specific statistics
        for split_name, split_df in splits.items():
            class_dist = split_df['label'].value_counts().to_dict()
            patient_count = split_df['patient_id'].nunique()
            study_count = split_df.groupby(['patient_id', 'study_name']).size().count()
            
            final_stats['split_statistics'][split_name] = {
                'total_images': len(split_df),
                'total_patients': patient_count,
                'total_studies': study_count,
                'class_distribution': class_dist,
                'images_per_patient_avg': len(split_df) / patient_count if patient_count > 0 else 0
            }
        
        # Save statistics
        stats_file = get_metadata_file('dataset_organization_stats.json')
        with open(stats_file, 'w') as f:
            # Convert Counter objects to dict for JSON serialization
            stats_to_save = self._convert_for_json(final_stats)
            json.dump(stats_to_save, f, indent=2)
        
        logger.info(f"Statistics report saved to: {stats_file}")
        
        return final_stats
    
    def generate_summary_report(self, final_stats):
        """Generate human-readable summary report."""
        logger.info("Generating summary report...")
        
        report_lines = []
        report_lines.append("DATASET ORGANIZATION SUMMARY REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Configuration
        config = final_stats['configuration']
        report_lines.append("CONFIGURATION:")
        report_lines.append(f"  Train/Validation/Test Ratios: {config['train_ratio']}/{config['validation_ratio']}/{config['test_ratio']}")
        report_lines.append(f"  Patient-level Split: {config['patient_level_split']}")
        report_lines.append(f"  Stratified Splits: {config['stratify_splits']}")
        report_lines.append(f"  Focus Study Type: {config['focus_study_type']}")
        report_lines.append(f"  Random Seed: {config['random_seed']}")
        report_lines.append("")
        
        # Source statistics
        source_stats = final_stats['source_statistics']
        report_lines.append("SOURCE DATASET:")
        report_lines.append(f"  Total Images: {source_stats['total_images']}")
        report_lines.append(f"  Total Patients: {source_stats['total_patients']}")
        report_lines.append(f"  Total Studies: {source_stats['total_studies']}")
        report_lines.append(f"  Class Distribution: {dict(source_stats['class_distribution'])}")
        if source_stats['corrupted_files']:
            report_lines.append(f"  Corrupted Files: {len(source_stats['corrupted_files'])}")
        report_lines.append("")
        
        # Split statistics
        report_lines.append("FINAL SPLITS:")
        for split_name, split_stats in final_stats['split_statistics'].items():
            report_lines.append(f"  {split_name.upper()}:")
            report_lines.append(f"    Images: {split_stats['total_images']}")
            report_lines.append(f"    Patients: {split_stats['total_patients']}")
            report_lines.append(f"    Studies: {split_stats['total_studies']}")
            report_lines.append(f"    Class Distribution: {split_stats['class_distribution']}")
            report_lines.append(f"    Avg Images/Patient: {split_stats['images_per_patient_avg']:.1f}")
            report_lines.append("")
        
        # Copy statistics
        copy_stats = final_stats['copy_statistics']
        report_lines.append("COPY OPERATIONS:")
        report_lines.append(f"  Total Files Copied: {copy_stats['total_copied']}")
        report_lines.append(f"  Copy Errors: {len(copy_stats['copy_errors'])}")
        report_lines.append("")
        
        # File locations
        report_lines.append("OUTPUT LOCATIONS:")
        report_lines.append(f"  Organized Dataset: {final_stats['target_directory']}")
        report_lines.append(f"  Metadata Files: {METADATA_PATH}")
        report_lines.append("")
        
        # Save report
        report_file = get_report_file('dataset_organization_report.txt')
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Summary report saved to: {report_file}")
        
        # Also print to console
        print('\n'.join(report_lines))
        
        return report_file
    
    def extract_label_from_study_name(self, study_name):
        """Extract label from study directory name."""
        study_lower = study_name.lower()
        if 'negative' in study_lower:
            return 'negative'
        elif 'positive' in study_lower:
            return 'positive'
        else:
            logger.warning(f"Could not extract label from study name: {study_name}, defaulting to 'unknown'")
            return 'unknown'
    
    def is_valid_image_file(self, file_path):
        """Check if file is a valid image file."""
        return file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FORMATS
    
    def validate_image_file(self, file_path):
        """Validate that image file can be opened."""
        from PIL import Image
        with Image.open(file_path) as img:
            # Try to load the image data
            img.verify()
    
    def _convert_for_json(self, obj):
        """Convert objects for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, Counter):
            return dict(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64, np.floating)):
            return float(obj)
        else:
            return obj

def main():
    """Main function to organize dataset."""
    logger.info("Starting dataset organization...")
    
    try:
        # Initialize organizer
        organizer = DatasetOrganizer()
        
        # Step 1: Scan source directory
        logger.info("Step 1: Scanning source directory...")
        organizer.scan_source_directory()
        
        if organizer.stats['total_images'] == 0:
            logger.error("No valid images found in source directory!")
            return
        
        # Step 2: Create splits
        logger.info("Step 2: Creating train/validation/test splits...")
        splits = organizer.create_splits()
        
        # Step 3: Organize files
        logger.info("Step 3: Organizing files...")
        copy_stats = organizer.organize_files(splits)
        
        # Step 4: Generate metadata
        logger.info("Step 4: Generating metadata files...")
        metadata_files = organizer.generate_metadata_files(splits)
        
        # Step 5: Generate class mapping
        logger.info("Step 5: Generating class mapping...")
        class_mapping_file = organizer.generate_class_mapping()
        
        # Step 6: Generate statistics
        logger.info("Step 6: Generating statistics report...")
        final_stats = organizer.generate_statistics_report(splits, copy_stats)
        
        # Step 7: Generate summary report
        logger.info("Step 7: Generating summary report...")
        summary_report = organizer.generate_summary_report(final_stats)
        
        logger.info("Dataset organization completed successfully!")
        
        # Print quick summary
        print(f"\n{'='*60}")
        print(f"DATASET ORGANIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"✅ Total images processed: {organizer.stats['total_images']}")
        print(f"✅ Total files copied: {copy_stats['total_copied']}")
        print(f"✅ Organized dataset location: {organizer.target_path}")
        print(f"✅ Metadata files: {len(metadata_files)}")
        
        if copy_stats['copy_errors']:
            print(f"⚠️  Copy errors: {len(copy_stats['copy_errors'])}")
        
        print(f"\n📁 Key outputs:")
        print(f"   - Organized data: {organizer.target_path}")
        print(f"   - Metadata: {METADATA_PATH}")
        print(f"   - Reports: {REPORTS_PATH}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        logger.error(f"Error during dataset organization: {e}")
        raise

if __name__ == "__main__":
    main()