# Jupyter Notebook: 01_data_exploration.ipynb
# Interactive Data Exploration for Bone Fracture Detection

# Cell 1: Setup and Imports
"""
# Data Exploration Notebook
This notebook provides interactive analysis of the bone fracture dataset.
Run each cell sequentially to explore your data.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Custom modules
from config import *
from data_exploration import DatasetExplorer

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

print("✅ All packages imported successfully!")
print(f"📁 Data path: {RAW_DATA_PATH}")
print(f"🎯 Focus study type: {FOCUS_STUDY_TYPE}")

# Cell 2: Initialize Explorer and Quick Check
# Initialize the dataset explorer
explorer = DatasetExplorer()

# Quick check if data exists
if not RAW_DATA_PATH.exists():
    print("❌ ERROR: Data path does not exist!")
    print(f"Expected path: {RAW_DATA_PATH}")
    print("Please check your data directory structure.")
else:
    print("✅ Data directory found!")
    
    # List available study types
    study_types = [d.name for d in RAW_DATA_PATH.iterdir() if d.is_dir()]
    print(f"📊 Available study types: {study_types}")

# Cell 3: Dataset Structure Analysis
# Run comprehensive structure analysis
print("🔍 Starting dataset structure analysis...")
structure_stats = explorer.explore_dataset_structure()

# Display results
print(f"\n📈 DATASET OVERVIEW")
print(f"{'='*40}")
print(f"Total Patients: {structure_stats['total_patients']}")
print(f"Total Studies: {structure_stats['total_studies']}")
print(f"Total Images: {structure_stats['total_images']}")
print(f"Study Types: {structure_stats['study_types']}")

print(f"\n🏷️ LABEL DISTRIBUTION")
print(f"{'='*40}")
for label, count in structure_stats['label_distribution'].items():
    percentage = (count / structure_stats['total_studies']) * 100
    print(f"{label:>10}: {count:>6} ({percentage:>5.1f}%)")

print(f"\n👥 PATIENT STATISTICS")
print(f"{'='*40}")
if structure_stats['images_per_patient']:
    images_per_patient = structure_stats['images_per_patient']
    print(f"Avg images per patient: {np.mean(images_per_patient):.1f}")
    print(f"Min images per patient: {min(images_per_patient)}")
    print(f"Max images per patient: {max(images_per_patient)}")

# Cell 4: Interactive Visualizations - Dataset Overview
# Create interactive plots using Plotly

# 1. Label Distribution Pie Chart
if structure_stats['label_distribution']:
    labels = list(structure_stats['label_distribution'].keys())
    values = list(structure_stats['label_distribution'].values())
    
    fig_pie = px.pie(
        values=values, 
        names=labels, 
        title="Label Distribution in Dataset",
        color_discrete_sequence=['#FF6B6B', '#4ECDC4']
    )
    fig_pie.show()

# 2. Images per Patient Histogram
if structure_stats['images_per_patient']:
    fig_hist = px.histogram(
        x=structure_stats['images_per_patient'],
        nbins=20,
        title="Distribution of Images per Patient",
        labels={'x': 'Number of Images', 'y': 'Number of Patients'}
    )
    fig_hist.show()

# 3. Study Type Distribution
study_type_counts = {}
for study_type, studies in explorer.study_info.items():
    study_type_counts[study_type] = len(studies)

if study_type_counts:
    fig_bar = px.bar(
        x=list(study_type_counts.keys()),
        y=list(study_type_counts.values()),
        title="Number of Studies by Type",
        labels={'x': 'Study Type', 'y': 'Number of Studies'}
    )
    fig_bar.show()

# Cell 5: Sample Images Display
# Display sample images from each class
def display_sample_images(num_samples=5):
    """Display sample images from each class."""
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    fig.suptitle('Sample Images from Dataset', fontsize=16)
    
    # Get sample images for each label
    label_samples = {'negative': [], 'positive': []}
    
    for image_info in explorer.image_info:
        label = image_info['label']
        if label in label_samples and len(label_samples[label]) < num_samples:
            label_samples[label].append(image_info['file_path'])
    
    # Display images
    for i, label in enumerate(['negative', 'positive']):
        for j, image_path in enumerate(label_samples[label][:num_samples]):
            if j < num_samples:
                try:
                    img = Image.open(image_path)
                    axes[i, j].imshow(img, cmap='gray')
                    axes[i, j].set_title(f'{label.capitalize()} - Sample {j+1}')
                    axes[i, j].axis('off')
                except Exception as e:
                    axes[i, j].text(0.5, 0.5, f'Error loading\n{e}', 
                                  ha='center', va='center', transform=axes[i, j].transAxes)
                    axes[i, j].set_title(f'{label.capitalize()} - Error')
    
    plt.tight_layout()
    plt.show()

print("🖼️ Analyzing sample images (this may take a moment)...")
# Run image analysis on a sample
explorer.analyze_images(sample_size=200)  # Analyze first 200 images

# Display sample images
display_sample_images()

# Cell 6: Image Quality Analysis
# Analyze image characteristics
if 'images' in explorer.stats and explorer.stats['images']['dimensions']:
    
    # Create comprehensive image analysis plots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=['Image Dimensions', 'File Sizes (MB)', 'Brightness Distribution',
                       'Contrast Distribution', 'Aspect Ratios', 'Format Distribution']
    )
    
    # 1. Image dimensions scatter plot
    widths, heights = zip(*explorer.stats['images']['dimensions'])
    fig.add_trace(
        go.Scatter(x=widths, y=heights, mode='markers', name='Dimensions',
                  marker=dict(size=4, opacity=0.6)),
        row=1, col=1
    )
    
    # 2. File sizes histogram
    file_sizes_mb = [size / (1024 * 1024) for size in explorer.stats['images']['file_sizes']]
    fig.add_trace(
        go.Histogram(x=file_sizes_mb, name='File Sizes', nbinsx=30),
        row=1, col=2
    )
    
    # 3. Brightness histogram
    fig.add_trace(
        go.Histogram(x=explorer.stats['images']['brightness'], name='Brightness', nbinsx=30),
        row=1, col=3
    )
    
    # 4. Contrast histogram
    fig.add_trace(
        go.Histogram(x=explorer.stats['images']['contrast'], name='Contrast', nbinsx=30),
        row=2, col=1
    )
    
    # 5. Aspect ratios histogram
    fig.add_trace(
        go.Histogram(x=explorer.stats['images']['aspect_ratios'], name='Aspect Ratios', nbinsx=20),
        row=2, col=2
    )
    
    # 6. Format distribution
    format_counts = dict(explorer.stats['images']['formats'])
    fig.add_trace(
        go.Bar(x=list(format_counts.keys()), y=list(format_counts.values()), name='Formats'),
        row=2, col=3
    )
    
    fig.update_layout(height=800, showlegend=False, title_text="Image Quality Analysis")
    fig.show()

# Cell 7: Data Quality Assessment
# Check for potential issues
print("🔍 DATA QUALITY ASSESSMENT")
print("="*50)

if 'images' in explorer.stats:
    img_stats = explorer.stats['images']
    
    # Check for corrupted files
    if img_stats['corrupted_files']:
        print(f"⚠️  Found {len(img_stats['corrupted_files'])} corrupted files:")
        for corrupted_file in img_stats['corrupted_files'][:10]:  # Show first 10
            print(f"   - {corrupted_file}")
        if len(img_stats['corrupted_files']) > 10:
            print(f"   ... and {len(img_stats['corrupted_files']) - 10} more")
    else:
        print("✅ No corrupted files found")
    
    # Check image dimensions consistency
    if img_stats['dimensions']:
        widths, heights = zip(*img_stats['dimensions'])
        unique_dimensions = len(set(img_stats['dimensions']))
        print(f"\n📐 Dimension Analysis:")
        print(f"   Unique dimensions: {unique_dimensions}")
        print(f"   Width range: {min(widths)} - {max(widths)} pixels")
        print(f"   Height range: {min(heights)} - {max(heights)} pixels")
        
        if unique_dimensions > 10:
            print("   ⚠️  High dimension variability - consider standardization")
        else:
            print("   ✅ Reasonable dimension consistency")
    
    # Check file sizes
    if img_stats['file_sizes']:
        avg_size_mb = np.mean([size / (1024 * 1024) for size in img_stats['file_sizes']])
        max_size_mb = max(img_stats['file_sizes']) / (1024 * 1024)
        print(f"\n💾 File Size Analysis:")
        print(f"   Average size: {avg_size_mb:.2f} MB")
        print(f"   Maximum size: {max_size_mb:.2f} MB")
        
        if max_size_mb > 50:
            print("   ⚠️  Large files detected - consider compression")
        else:
            print("   ✅ File sizes are reasonable")

# Cell 8: Class Balance Analysis
# Analyze class distribution and balance
print("⚖️  CLASS BALANCE ANALYSIS")
print("="*50)

if 'structure' in explorer.stats:
    label_dist = explorer.stats['structure']['label_distribution']
    total_studies = sum(label_dist.values())
    
    print(f"Total studies: {total_studies}")
    print(f"Class distribution:")
    
    for label, count in label_dist.items():
        percentage = (count / total_studies) * 100
        print(f"  {label:>10}: {count:>6} ({percentage:>5.1f}%)")
    
    # Calculate class imbalance ratio
    if len(label_dist) == 2:
        counts = list(label_dist.values())
        imbalance_ratio = max(counts) / min(counts)
        print(f"\nImbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 2:
            print("⚠️  Significant class imbalance detected")
            print("   Consider using stratified sampling and class weights")
        elif imbalance_ratio > 1.5:
            print("⚠️  Moderate class imbalance detected")
            print("   Monitor model performance on minority class")
        else:
            print("✅ Classes are reasonably balanced")

# Cell 9: Patient-Level Analysis
# Analyze data at patient level for proper train/test splitting
print("👤 PATIENT-LEVEL ANALYSIS")
print("="*50)

patient_label_dist = {}
patients_with_mixed_labels = []

for patient_id, patient_info in explorer.patient_info.items():
    # Get all studies for this patient
    patient_studies = []
    for study_type, studies in explorer.study_info.items():
        for study in studies:
            if study['patient_id'] == patient_id:
                patient_studies.append(study['label'])
    
    # Check if patient has mixed labels
    unique_labels = set(patient_studies)
    if len(unique_labels) > 1:
        patients_with_mixed_labels.append(patient_id)
    
    # Count patients by primary label
    primary_label = max(set(patient_studies), key=patient_studies.count)
    patient_label_dist[primary_label] = patient_label_dist.get(primary_label, 0) + 1

print(f"Patient distribution by primary label:")
for label, count in patient_label_dist.items():
    total_patients = sum(patient_label_dist.values())
    percentage = (count / total_patients) * 100
    print(f"  {label:>10}: {count:>6} ({percentage:>5.1f}%)")

print(f"\nPatients with mixed labels: {len(patients_with_mixed_labels)}")
if patients_with_mixed_labels:
    print("⚠️  Some patients have both positive and negative studies")
    print("   Consider patient-level splitting to avoid data leakage")
else:
    print("✅ No mixed labels per patient - good for patient-level splitting")

# Cell 10: Recommendations and Next Steps
print("🎯 RECOMMENDATIONS FOR PREPROCESSING")
print("="*50)

recommendations = []

# Data quality recommendations
if 'images' in explorer.stats:
    img_stats = explorer.stats['images']
    
    if img_stats['corrupted_files']:
        recommendations.append("🔧 Remove or fix corrupted image files")
    
    if 'summary' in img_stats:
        img_summary = img_stats['summary']
        
        # Dimension standardization
        if len(set(img_stats['dimensions'])) > 10:
            recommendations.append("📐 Standardize image dimensions (resize to 224x224 or 512x512)")
        
        # Brightness normalization
        if img_summary['brightness_stats']['std'] > 50:
            recommendations.append("🔆 Apply histogram equalization for brightness normalization")
        
        # File size optimization
        file_sizes_mb = [size / (1024 * 1024) for size in img_stats['file_sizes']]
        if max(file_sizes_mb) > 20:
            recommendations.append("💾 Compress large images to reduce storage and loading time")

# Class balance recommendations
if 'structure' in explorer.stats:
    label_dist = explorer.stats['structure']['label_distribution']
    if len(label_dist) == 2:
        counts = list(label_dist.values())
        imbalance_ratio = max(counts) / min(counts)
        if imbalance_ratio > 1.5:
            recommendations.append("⚖️ Address class imbalance with stratified sampling or class weights")

# Splitting recommendations
if patients_with_mixed_labels:
    recommendations.append("👤 Use patient-level splitting to prevent data leakage")
else:
    recommendations.append("✅ Patient-level splitting is safe to use")

# Data augmentation recommendations
recommendations.append("🔄 Apply data augmentation: rotation (±15°), brightness/contrast adjustment")
recommendations.append("📊 Use stratified splitting to maintain class balance")

# Print recommendations
for i, rec in enumerate(recommendations, 1):
    print(f"{i:>2}. {rec}")

print(f"\n🚀 NEXT STEPS:")
print("1. Run create_dataset_structure.py to organize your data")
print("2. Run data_preprocessing.py to clean and standardize images")
print("3. Run data_augmentation.py to generate additional training data")
print("4. Proceed to model training phase")

# Cell 11: Export Results
# Save exploration results
print("💾 EXPORTING RESULTS")
print("="*20)

try:
    # Export statistics
    stats_file = explorer.export_statistics()
    print(f"✅ Statistics exported to: {stats_file}")
    
    # Generate summary report
    report_file = explorer.generate_summary_report()
    print(f"✅ Summary report saved to: {report_file}")
    
    # Generate visualizations
    if GENERATE_HTML_REPORTS:
        fig = explorer.generate_visualizations()
        print("✅ Visualizations generated and saved")
    
    print(f"\n📁 All results saved to:")
    print(f"   Metadata: {METADATA_PATH}")
    print(f"   Reports: {REPORTS_PATH}")
    
except Exception as e:
    print(f"❌ Error exporting results: {e}")

print("\n🎉 DATA EXPLORATION COMPLETE!")
print("You can now proceed to the next phase: Data Preprocessing")