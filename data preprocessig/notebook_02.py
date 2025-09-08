# Jupyter Notebook: 02_image_preprocessing.ipynb
# Image Preprocessing Experiments and Testing

# Cell 1: Setup and Imports
"""
# Image Preprocessing Experiments
This notebook allows you to test different preprocessing techniques
and visualize their effects on sample images.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageEnhance
import cv2
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Custom modules
from config import *
from image_utils import ImageProcessor
from data_preprocessing import DataPreprocessor

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

print("✅ All packages imported successfully!")
print(f"📁 Data path: {PROCESSED_DATA_PATH}")

# Cell 2: Initialize Image Processor and Load Sample Images
# Initialize the image processor
processor = ImageProcessor()

# Find sample images from different classes
sample_images = {}
sample_paths = {}

for split in ['train', 'validation']:
    split_path = PROCESSED_DATA_PATH / split
    if not split_path.exists():
        continue
        
    for class_name in CLASS_NAMES:
        class_path = split_path / class_name
        if not class_path.exists():
            continue
            
        # Get first few images from this class
        image_files = [f for f in class_path.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']][:5]
        
        if image_files:
            key = f"{split}_{class_name}"
            sample_paths[key] = image_files
            sample_images[key] = []
            
            for img_path in image_files[:3]:  # Load first 3 images
                img = processor.load_image(img_path)
                if img is not None:
                    sample_images[key].append(img)

print(f"📊 Sample images loaded:")
for key, images in sample_images.items():
    print(f"  {key}: {len(images)} images")

# Cell 3: Test Different Preprocessing Techniques
def compare_preprocessing_techniques(image, title="Image Preprocessing Comparison"):
    """Compare different preprocessing techniques on a single image."""
    
    techniques = {
        'Original': image,
        'Resized': processor.resize_image(image, IMAGE_SIZE),
        'Normalized': processor.normalize_pixels(image.copy()),
        'CLAHE': processor.apply_clahe(image.copy()),
        'Enhanced Contrast': processor.enhance_contrast(image.copy(), factor=1.3),
        'Enhanced Brightness': processor.enhance_brightness(image.copy(), factor=1.2)
    }
    
    # Remove None results
    techniques = {k: v for k, v in techniques.items() if v is not None}
    
    # Create subplot
    n_techniques = len(techniques)
    cols = 3
    rows = (n_techniques + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    fig.suptitle(title, fontsize=16)
    
    # Flatten axes for easy indexing
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, (technique_name, processed_img) in enumerate(techniques.items()):
        if i < len(axes):
            # Display image
            if len(processed_img.shape) == 3:
                axes[i].imshow(processed_img)
            else:
                axes[i].imshow(processed_img, cmap='gray')
            
            axes[i].set_title(f'{technique_name}\nShape: {processed_img.shape}')
            axes[i].axis('off')
            
            # Add statistics
            stats = processor.calculate_image_statistics(processed_img)
            stats_text = f"Mean: {stats.get('mean', 0):.1f}\nStd: {stats.get('std', 0):.1f}"
            axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                        verticalalignment='top', fontsize=8)
    
    # Hide unused subplots
    for i in range(len(techniques), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Test preprocessing on sample images
if sample_images:
    for key, images in list(sample_images.items())[:2]:  # Show first 2 categories
        if images:
            print(f"\n🖼️ Testing preprocessing on {key}")
            compare_preprocessing_techniques(images[0], f"Preprocessing Techniques - {key}")

# Cell 4: Interactive Parameter Testing
from ipywidgets import interact, FloatSlider, IntSlider, Checkbox, Dropdown
import ipywidgets as widgets

# Select a sample image for interactive testing
if sample_images:
    # Get first available image
    test_image = None
    test_key = None
    for key, images in sample_images.items():
        if images:
            test_image = images[0]
            test_key = key
            break
    
    if test_image is not None:
        print(f"🎛️ Interactive preprocessing controls for {test_key}")
        
        def interactive_preprocessing(
            resize_width=224,
            resize_height=224,
            normalize=True,
            apply_clahe=True,
            clahe_clip_limit=2.0,
            contrast_factor=1.0,
            brightness_factor=1.0,
            add_noise=False,
            noise_variance=0.01
        ):
            # Start with original image
            result_image = test_image.copy()
            processing_steps = ["Original"]
            
            # Apply transformations
            if resize_width != test_image.shape[1] or resize_height != test_image.shape[0]:
                result_image = processor.resize_image(result_image, (resize_width, resize_height))
                processing_steps.append("Resized")
            
            if apply_clahe:
                result_image = processor.apply_clahe(result_image, clip_limit=clahe_clip_limit)
                processing_steps.append("CLAHE")
            
            if contrast_factor != 1.0:
                result_image = processor.enhance_contrast(result_image, factor=contrast_factor)
                processing_steps.append("Contrast")
            
            if brightness_factor != 1.0:
                result_image = processor.enhance_brightness(result_image, factor=brightness_factor)
                processing_steps.append("Brightness")
            
            if add_noise:
                result_image = processor.add_gaussian_noise(result_image, variance=noise_variance)
                processing_steps.append("Noise")
            
            if normalize:
                result_image = processor.normalize_pixels(result_image)
                processing_steps.append("Normalized")
            
            # Display results
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Original
            if len(test_image.shape) == 3:
                axes[0].imshow(test_image)
            else:
                axes[0].imshow(test_image, cmap='gray')
            axes[0].set_title(f'Original\n{test_image.shape}')
            axes[0].axis('off')
            
            # Processed
            if result_image is not None:
                if len(result_image.shape) == 3:
                    axes[1].imshow(result_image)
                else:
                    axes[1].imshow(result_image, cmap='gray')
                axes[1].set_title(f'Processed\n{result_image.shape}\nSteps: {", ".join(processing_steps)}')
                axes[1].axis('off')
                
                # Show statistics
                stats = processor.calculate_image_statistics(result_image)
                stats_text = (f"Mean: {stats.get('mean', 0):.1f}\n"
                             f"Std: {stats.get('std', 0):.1f}\n"
                             f"Min: {stats.get('min', 0):.1f}\n"
                             f"Max: {stats.get('max', 0):.1f}")
                axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                           verticalalignment='top', fontsize=9)
            
            plt.tight_layout()
            plt.show()
        
        # Create interactive widget (Note: This works in Jupyter notebooks)
        interact(
            interactive_preprocessing,
            resize_width=IntSlider(min=64, max=512, step=32, value=224),
            resize_height=IntSlider(min=64, max=512, step=32, value=224),
            normalize=Checkbox(value=True),
            apply_clahe=Checkbox(value=True),
            clahe_clip_limit=FloatSlider(min=0.5, max=5.0, step=0.5, value=2.0),
            contrast_factor=FloatSlider(min=0.5, max=2.0, step=0.1, value=1.0),
            brightness_factor=FloatSlider(min=0.5, max=2.0, step=0.1, value=1.0),
            add_noise=Checkbox(value=False),
            noise_variance=FloatSlider(min=0.001, max=0.05, step=0.001, value=0.01)
        )

# Cell 5: Quality Assessment Testing
def analyze_image_quality(images_dict, title="Image Quality Analysis"):
    """Analyze quality metrics across different images."""
    
    quality_data = []
    
    for category, images in images_dict.items():
        for i, image in enumerate(images):
            stats = processor.calculate_image_statistics(image)
            is_valid, issues = processor.validate_image_quality(image)
            
            quality_data.append({
                'category': category,
                'image_idx': i,
                'mean_brightness': stats.get('mean', 0),
                'contrast': stats.get('contrast', 0),
                'width': stats.get('width', 0),
                'height': stats.get('height', 0),
                'is_valid': is_valid,
                'num_issues': len(issues),
                'too_dark': stats.get('too_dark', False),
                'too_bright': stats.get('too_bright', False),
                'low_contrast': stats.get('low_contrast', False)
            })
    
    df = pd.DataFrame(quality_data)
    
    if len(df) > 0:
        # Create quality analysis plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Brightness Distribution', 'Contrast Distribution', 
                           'Quality Issues by Category', 'Dimension Analysis']
        )
        
        # Brightness distribution
        for category in df['category'].unique():
            category_data = df[df['category'] == category]
            fig.add_trace(
                go.Histogram(x=category_data['mean_brightness'], name=f'{category} Brightness',
                           opacity=0.7, nbinsx=20),
                row=1, col=1
            )
        
        # Contrast distribution
        for category in df['category'].unique():
            category_data = df[df['category'] == category]
            fig.add_trace(
                go.Histogram(x=category_data['contrast'], name=f'{category} Contrast',
                           opacity=0.7, nbinsx=20),
                row=1, col=2
            )
        
        # Quality issues
        issue_summary = df.groupby('category')['num_issues'].mean().reset_index()
        fig.add_trace(
            go.Bar(x=issue_summary['category'], y=issue_summary['num_issues'],
                  name='Avg Issues per Image'),
            row=2, col=1
        )
        
        # Dimensions scatter
        fig.add_trace(
            go.Scatter(x=df['width'], y=df['height'], mode='markers',
                      color=df['category'], text=df['category'],
                      name='Dimensions'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text=title, showlegend=True)
        fig.show()
        
        # Show summary statistics
        print(f"\n📊 QUALITY ANALYSIS SUMMARY")
        print(f"{'='*40}")
        print(f"Total images analyzed: {len(df)}")
        print(f"Valid images: {df['is_valid'].sum()}/{len(df)} ({df['is_valid'].mean():.1%})")
        print(f"Images too dark: {df['too_dark'].sum()}")
        print(f"Images too bright: {df['too_bright'].sum()}")
        print(f"Images with low contrast: {df['low_contrast'].sum()}")
        
        print(f"\n📈 STATISTICS BY CATEGORY:")
        for category in df['category'].unique():
            cat_data = df[df['category'] == category]
            print(f"  {category}:")
            print(f"    Valid: {cat_data['is_valid'].sum()}/{len(cat_data)}")
            print(f"    Avg brightness: {cat_data['mean_brightness'].mean():.1f}")
            print(f"    Avg contrast: {cat_data['contrast'].mean():.1f}")
    
    return df

# Run quality analysis on sample images
if sample_images:
    quality_df = analyze_image_quality(sample_images, "Sample Images Quality Analysis")

# Cell 6: Preprocessing Pipeline Testing
def test_full_preprocessing_pipeline(image_paths, num_samples=5):
    """Test the full preprocessing pipeline on sample images."""
    
    if isinstance(image_paths, dict):
        # Flatten the image paths from dictionary
        all_paths = []
        for paths_list in image_paths.values():
            all_paths.extend(paths_list)
        image_paths = all_paths
    
    # Limit to specified number of samples
    sample_paths = image_paths[:num_samples]
    
    # Test preprocessing pipeline
    results = []
    
    for img_path in sample_paths:
        print(f"🔄 Processing: {img_path.name}")
        
        # Run full preprocessing pipeline
        processed_img, processing_info = processor.preprocess_pipeline(img_path)
        
        results.append({
            'path': img_path,
            'processed_image': processed_img,
            'processing_info': processing_info,
            'success': processing_info['success']
        })
    
    # Create visualization of results
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        fig, axes = plt.subplots(len(successful_results), 3, figsize=(15, 5 * len(successful_results)))
        if len(successful_results) == 1:
            axes = axes.reshape(1, -1)
        
        for i, result in enumerate(successful_results):
            # Load original
            original = processor.load_image(result['path'])
            processed = result['processed_image']
            info = result['processing_info']
            
            # Original image
            if original is not None:
                axes[i, 0].imshow(original, cmap='gray' if len(original.shape) == 2 else None)
                axes[i, 0].set_title(f'Original\n{original.shape}')
                axes[i, 0].axis('off')
            
            # Processed image
            if processed is not None:
                axes[i, 1].imshow(processed, cmap='gray' if len(processed.shape) == 2 else None)
                axes[i, 1].set_title(f'Processed\n{processed.shape}')
                axes[i, 1].axis('off')
            
            # Processing information
            axes[i, 2].axis('off')
            info_text = f"PROCESSING INFO:\n\n"
            info_text += f"Steps: {', '.join(info['steps_applied'])}\n\n"
            info_text += f"Quality Issues: {len(info['quality_issues'])}\n"
            if info['quality_issues']:
                info_text += f"Issues:\n"
                for issue in info['quality_issues'][:3]:  # Show first 3 issues
                    info_text += f"  • {issue}\n"
            
            if 'original_stats' in info:
                orig_stats = info['original_stats']
                info_text += f"\nORIGINAL:\n"
                info_text += f"  Mean: {orig_stats.get('mean', 0):.1f}\n"
                info_text += f"  Std: {orig_stats.get('std', 0):.1f}\n"
            
            if 'final_stats' in info:
                final_stats = info['final_stats']
                info_text += f"\nFINAL:\n"
                info_text += f"  Mean: {final_stats.get('mean', 0):.1f}\n"
                info_text += f"  Std: {final_stats.get('std', 0):.1f}\n"
            
            axes[i, 2].text(0.05, 0.95, info_text, transform=axes[i, 2].transAxes,
                           fontsize=9, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('Full Preprocessing Pipeline Results', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    # Print summary
    success_count = len(successful_results)
    total_count = len(results)
    
    print(f"\n📊 PIPELINE TESTING SUMMARY")
    print(f"{'='*40}")
    print(f"Successful: {success_count}/{total_count} ({success_count/total_count:.1%})")
    
    if success_count < total_count:
        failed_results = [r for r in results if not r['success']]
        print(f"Failed images:")
        for result in failed_results:
            error = result['processing_info'].get('error', 'Unknown error')
            print(f"  • {result['path'].name}: {error}")
    
    return results

# Test preprocessing pipeline on sample images
if sample_paths:
    # Get some sample paths for testing
    test_paths = []
    for paths_list in list(sample_paths.values())[:2]:  # First 2 categories
        test_paths.extend(paths_list[:2])  # 2 images from each category
    
    if test_paths:
        pipeline_results = test_full_preprocessing_pipeline(test_paths)

# Cell 7: Batch Processing Performance Test
def test_batch_processing_performance(image_paths, batch_sizes=[1, 5, 10]):
    """Test preprocessing performance with different batch sizes."""
    
    import time
    from tqdm import tqdm
    
    performance_results = []
    
    # Limit to reasonable number for testing
    test_paths = image_paths[:20] if len(image_paths) > 20 else image_paths
    
    print(f"🚀 Testing batch processing performance on {len(test_paths)} images")
    
    for batch_size in batch_sizes:
        print(f"\n📦 Testing batch size: {batch_size}")
        
        start_time = time.time()
        processed_count = 0
        error_count = 0
        
        # Process in batches
        for i in range(0, len(test_paths), batch_size):
            batch_paths = test_paths[i:i + batch_size]
            
            for img_path in batch_paths:
                try:
                    processed_img, info = processor.preprocess_pipeline(img_path)
                    if info['success']:
                        processed_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    error_count += 1
        
        total_time = time.time() - start_time
        avg_time_per_image = total_time / len(test_paths)
        
        result = {
            'batch_size': batch_size,
            'total_time': total_time,
            'avg_time_per_image': avg_time_per_image,
            'images_per_second': len(test_paths) / total_time,
            'processed_count': processed_count,
            'error_count': error_count,
            'success_rate': processed_count / len(test_paths)
        }
        
        performance_results.append(result)
        
        print(f"  ✅ Time: {total_time:.2f}s")
        print(f"  ✅ Avg per image: {avg_time_per_image:.3f}s")
        print(f"  ✅ Images/second: {result['images_per_second']:.2f}")
        print(f"  ✅ Success rate: {result['success_rate']:.1%}")
    
    # Visualize performance results
    df_perf = pd.DataFrame(performance_results)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Processing time comparison
    axes[0].bar(df_perf['batch_size'], df_perf['avg_time_per_image'])
    axes[0].set_title('Average Processing Time per Image')
    axes[0].set_xlabel('Batch Size')
    axes[0].set_ylabel('Time (seconds)')
    
    # Add value labels on bars
    for i, row in df_perf.iterrows():
        axes[0].text(row['batch_size'], row['avg_time_per_image'] + 0.001, 
                    f"{row['avg_time_per_image']:.3f}s", ha='center')
    
    # Throughput comparison
    axes[1].bar(df_perf['batch_size'], df_perf['images_per_second'], color='orange')
    axes[1].set_title('Processing Throughput')
    axes[1].set_xlabel('Batch Size')
    axes[1].set_ylabel('Images per Second')
    
    # Add value labels on bars
    for i, row in df_perf.iterrows():
        axes[1].text(row['batch_size'], row['images_per_second'] + 0.1, 
                    f"{row['images_per_second']:.2f}", ha='center')
    
    plt.tight_layout()
    plt.show()
    
    return performance_results

# Test batch processing performance
if sample_paths:
    # Get sample paths for performance testing
    perf_test_paths = []
    for paths_list in sample_paths.values():
        perf_test_paths.extend(paths_list[:5])  # 5 images from each category
    
    if len(perf_test_paths) > 5:
        performance_results = test_batch_processing_performance(perf_test_paths)

# Cell 8: Memory Usage Analysis
def analyze_memory_usage():
    """Analyze memory usage during preprocessing."""
    
    import psutil
    import gc
    
    print("🧠 MEMORY USAGE ANALYSIS")
    print("="*40)
    
    # Get current process
    process = psutil.Process(os.getpid())
    
    # Initial memory
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Test memory usage with different image sizes
    test_sizes = [(224, 224), (512, 512), (1024, 1024)]
    memory_usage = []
    
    for size in test_sizes:
        # Create test images
        test_images = []
        
        # Load some sample images and resize them
        if sample_images:
            sample_img = list(sample_images.values())[0][0]  # Get first available image
            
            gc.collect()  # Clear memory
            mem_before = process.memory_info().rss / 1024 / 1024
            
            # Create multiple images of this size
            for i in range(10):
                resized = processor.resize_image(sample_img.copy(), size)
                if resized is not None:
                    test_images.append(resized)
            
            mem_after = process.memory_info().rss / 1024 / 1024
            mem_diff = mem_after - mem_before
            
            memory_usage.append({
                'size': f"{size[0]}x{size[1]}",
                'width': size[0],
                'height': size[1],
                'memory_mb': mem_diff,
                'memory_per_image': mem_diff / len(test_images) if test_images else 0
            })
            
            print(f"Size {size[0]}x{size[1]}: {mem_diff:.2f} MB for {len(test_images)} images")
            
            # Clean up
            del test_images
            gc.collect()
    
    # Visualize memory usage
    if memory_usage:
        df_mem = pd.DataFrame(memory_usage)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.bar(df_mem['size'], df_mem['memory_mb'])
        plt.title('Total Memory Usage by Image Size')
        plt.xlabel('Image Size')
        plt.ylabel('Memory Usage (MB)')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        plt.bar(df_mem['size'], df_mem['memory_per_image'], color='orange')
        plt.title('Memory Usage per Image')
        plt.xlabel('Image Size')
        plt.ylabel('Memory per Image (MB)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Calculate optimal batch size based on available memory
        available_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024  # GB
        print(f"\n💾 Available memory: {available_memory:.2f} GB")
        
        for _, row in df_mem.iterrows():
            if row['memory_per_image'] > 0:
                max_batch_size = int((available_memory * 1024 * 0.5) / row['memory_per_image'])  # Use 50% of available memory
                print(f"Recommended max batch size for {row['size']}: {max_batch_size}")

# Run memory analysis
analyze_memory_usage()

# Cell 9: Configuration Optimization
def suggest_optimal_configuration():
    """Suggest optimal preprocessing configuration based on analysis."""
    
    print("🎯 OPTIMAL CONFIGURATION SUGGESTIONS")
    print("="*50)
    
    suggestions = []
    
    # Analyze image dimensions from quality analysis
    if 'quality_df' in locals() and len(quality_df) > 0:
        avg_width = quality_df['width'].mean()
        avg_height = quality_df['height'].mean()
        
        print(f"📐 DIMENSION ANALYSIS:")
        print(f"  Average original size: {avg_width:.0f}x{avg_height:.0f}")
        
        # Suggest target size
        if avg_width > 512 or avg_height > 512:
            suggestions.append("Consider using 512x512 target size for better quality")
            suggested_size = (512, 512)
        else:
            suggestions.append("224x224 target size should be sufficient")
            suggested_size = (224, 224)
        
        print(f"  Suggested target size: {suggested_size[0]}x{suggested_size[1]}")
        
        # Analyze brightness and contrast
        avg_brightness = quality_df['mean_brightness'].mean()
        avg_contrast = quality_df['contrast'].mean()
        
        print(f"\n🔆 QUALITY ANALYSIS:")
        print(f"  Average brightness: {avg_brightness:.1f}")
        print(f"  Average contrast: {avg_contrast:.1f}")
        
        if avg_brightness < 100:
            suggestions.append("Images appear dark - enable brightness enhancement")
        elif avg_brightness > 180:
            suggestions.append("Images appear bright - consider brightness normalization")
        
        if avg_contrast < 40:
            suggestions.append("Low contrast detected - enable CLAHE")
            suggestions.append("Consider contrast enhancement factor of 1.2-1.3")
        
        # Check for quality issues
        issue_rate = (quality_df['num_issues'] > 0).mean()
        print(f"  Images with quality issues: {issue_rate:.1%}")
        
        if issue_rate > 0.2:
            suggestions.append("High rate of quality issues - implement quality filtering")
    
    # Performance suggestions
    if 'performance_results' in locals():
        best_performance = min(performance_results, key=lambda x: x['avg_time_per_image'])
        print(f"\n⚡ PERFORMANCE ANALYSIS:")
        print(f"  Best batch size: {best_performance['batch_size']}")
        print(f"  Optimal processing time: {best_performance['avg_time_per_image']:.3f}s per image")
        
        if best_performance['avg_time_per_image'] > 0.1:
            suggestions.append("Consider optimizing preprocessing pipeline for speed")
    
    # Memory suggestions
    print(f"\n💾 MEMORY RECOMMENDATIONS:")
    suggestions.append(f"Use batch size of {BATCH_SIZE} for training")
    suggestions.append("Enable garbage collection between batches for large datasets")
    
    # Final recommendations
    print(f"\n✅ FINAL RECOMMENDATIONS:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. {suggestion}")
    
    # Generate optimal config
    optimal_config = {
        'IMAGE_SIZE': suggested_size if 'suggested_size' in locals() else IMAGE_SIZE,
        'APPLY_CLAHE': True,
        'CLAHE_CLIP_LIMIT': 2.5 if 'avg_contrast' in locals() and avg_contrast < 40 else 2.0,
        'NORMALIZE_PIXELS': True,
        'RESIZE_METHOD': 'bilinear',
        'BATCH_SIZE': 32 if 'suggested_size' not in locals() or suggested_size[0] <= 224 else 16
    }
    
    print(f"\n⚙️ SUGGESTED CONFIG UPDATES:")
    for key, value in optimal_config.items():
        current_value = globals().get(key, "Not set")
        if current_value != value:
            print(f"  {key}: {current_value} → {value}")

# Generate configuration suggestions
suggest_optimal_configuration()

# Cell 10: Export Results and Create Report
def create_preprocessing_experiment_report():
    """Create a comprehensive report of preprocessing experiments."""
    
    report_data = {
        'experiment_timestamp': pd.Timestamp.now().isoformat(),
        'configuration_tested': {
            'IMAGE_SIZE': IMAGE_SIZE,
            'APPLY_CLAHE': APPLY_CLAHE,
            'NORMALIZE_PIXELS': NORMALIZE_PIXELS,
            'CLAHE_CLIP_LIMIT': CLAHE_CLIP_LIMIT
        },
        'samples_analyzed': {},
        'quality_analysis': {},
        'performance_analysis': {},
        'recommendations': []
    }
    
    # Add sample information
    if 'sample_images' in locals():
        report_data['samples_analyzed'] = {
            category: len(images) for category, images in sample_images.items()
        }
    
    # Add quality analysis
    if 'quality_df' in locals() and len(quality_df) > 0:
        report_data['quality_analysis'] = {
            'total_images': len(quality_df),
            'valid_images': int(quality_df['is_valid'].sum()),
            'avg_brightness': float(quality_df['mean_brightness'].mean()),
            'avg_contrast': float(quality_df['contrast'].mean()),
            'quality_issues_rate': float((quality_df['num_issues'] > 0).mean())
        }
    
    # Add performance analysis
    if 'performance_results' in locals():
        report_data['performance_analysis'] = performance_results
    
    # Save report
    report_file = get_report_file('preprocessing_experiments_report.json')
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"📄 Experiment report saved: {report_file}")
    
    # Create summary visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Preprocessing Experiments Summary', fontsize=16)
    
    # Sample distribution
    if report_data['samples_analyzed']:
        categories = list(report_data['samples_analyzed'].keys())
        counts = list(report_data['samples_analyzed'].values())
        
        axes[0, 0].bar(categories, counts)
        axes[0, 0].set_title('Samples Analyzed by Category')
        axes[0, 0].set_ylabel('Number of Images')
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Quality metrics
    if 'quality_df' in locals() and len(quality_df) > 0:
        quality_df.boxplot(column='mean_brightness', by='category', ax=axes[0, 1])
        axes[0, 1].set_title('Brightness Distribution by Category')
        axes[0, 1].set_xlabel('Category')
        axes[0, 1].set_ylabel('Mean Brightness')
    
    # Performance comparison
    if 'performance_results' in locals():
        perf_df = pd.DataFrame(performance_results)
        axes[1, 0].plot(perf_df['batch_size'], perf_df['avg_time_per_image'], 'o-')
        axes[1, 0].set_title('Processing Time vs Batch Size')
        axes[1, 0].set_xlabel('Batch Size')
        axes[1, 0].set_ylabel('Avg Time per Image (s)')
    
    # Configuration summary (text)
    config_text = "CURRENT CONFIGURATION:\n\n"
    config_text += f"Image Size: {IMAGE_SIZE}\n"
    config_text += f"Apply CLAHE: {APPLY_CLAHE}\n"
    config_text += f"Normalize: {NORMALIZE_PIXELS}\n"
    config_text += f"CLAHE Clip: {CLAHE_CLIP_LIMIT}\n"
    
    axes[1, 1].text(0.1, 0.9, config_text, transform=axes[1, 1].transAxes,
                   fontsize=12, verticalalignment='top', fontfamily='monospace')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save summary plot
    summary_plot_file = get_report_file('preprocessing_experiments_summary.png')
    plt.savefig(summary_plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Summary plot saved: {summary_plot_file}")
    
    return report_file

# Create final report
experiment_report = create_preprocessing_experiment_report()

print(f"\n🎉 PREPROCESSING EXPERIMENTS COMPLETE!")
print(f"📁 Reports saved to: {REPORTS_PATH}")
print(f"\n📋 Next steps:")
print(f"  1. Review the experiment results and recommendations")
print(f"  2. Update your config.py file with optimal parameters")
print(f"  3. Run the main preprocessing pipeline with optimized settings")
print(f"  4. Proceed to data augmentation experiments")
            