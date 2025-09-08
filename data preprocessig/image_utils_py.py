"""
Image utility functions for bone fracture detection preprocessing.
Contains common image processing operations and validation functions.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Union, Optional, List
import warnings

from config import *

# Set up logging
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageProcessor:
    """Comprehensive image processing utilities for medical images."""
    
    def __init__(self):
        """Initialize the image processor."""
        self.stats = {
            'processed_count': 0,
            'error_count': 0,
            'processing_times': []
        }
    
    def load_image(self, image_path: Union[str, Path], mode: str = 'RGB') -> Optional[np.ndarray]:
        """
        Load image from file with error handling.
        
        Args:
            image_path: Path to image file
            mode: Image mode ('RGB', 'L', 'RGBA')
            
        Returns:
            numpy array of image or None if error
        """
        try:
            image_path = Path(image_path)
            
            # Check file exists and is valid size
            if not image_path.exists():
                logger.error(f"Image file not found: {image_path}")
                return None
            
            file_size = image_path.stat().st_size
            if file_size < MIN_FILE_SIZE:
                logger.warning(f"File too small ({file_size} bytes): {image_path}")
                return None
            
            if file_size > MAX_FILE_SIZE:
                logger.warning(f"File too large ({file_size} bytes): {image_path}")
                return None
            
            # Load image with PIL
            with Image.open(image_path) as img:
                # Convert to desired mode
                if img.mode != mode:
                    img = img.convert(mode)
                
                # Convert to numpy array
                img_array = np.array(img)
                
                # Validate image dimensions
                if len(img_array.shape) < 2:
                    logger.error(f"Invalid image dimensions: {img_array.shape}")
                    return None
                
                height, width = img_array.shape[:2]
                if width < MIN_IMAGE_SIZE[0] or height < MIN_IMAGE_SIZE[1]:
                    logger.warning(f"Image too small ({width}x{height}): {image_path}")
                    return None
                
                return img_array
                
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            self.stats['error_count'] += 1
            return None
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int], 
                    method: str = 'bilinear') -> np.ndarray:
        """
        Resize image to target dimensions.
        
        Args:
            image: Input image array
            target_size: (width, height) tuple
            method: Resize method ('bilinear', 'nearest', 'bicubic', 'lanczos')
            
        Returns:
            Resized image array
        """
        if image is None:
            return None
        
        try:
            # Convert numpy array back to PIL for high-quality resizing
            if len(image.shape) == 3:
                pil_img = Image.fromarray(image)
            else:
                pil_img = Image.fromarray(image, mode='L')
            
            # Map resize methods
            resize_methods = {
                'nearest': Image.NEAREST,
                'bilinear': Image.BILINEAR,
                'bicubic': Image.BICUBIC,
                'lanczos': Image.LANCZOS
            }
            
            resize_method = resize_methods.get(method, Image.BILINEAR)
            
            # Resize image
            resized_img = pil_img.resize(target_size, resize_method)
            
            # Convert back to numpy array
            return np.array(resized_img)
            
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return None
    
    def normalize_pixels(self, image: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize pixel values to [0, 1] range.
        
        Args:
            image: Input image array
            method: Normalization method ('minmax', 'zscore', 'robust')
            
        Returns:
            Normalized image array
        """
        if image is None:
            return None
        
        try:
            image_float = image.astype(np.float32)
            
            if method == 'minmax':
                # Min-max normalization to [0, 1]
                img_min = np.min(image_float)
                img_max = np.max(image_float)
                
                if img_max > img_min:
                    normalized = (image_float - img_min) / (img_max - img_min)
                else:
                    normalized = image_float / 255.0 if img_max <= 255 else image_float
                    
            elif method == 'zscore':
                # Z-score normalization
                mean = np.mean(image_float)
                std = np.std(image_float)
                if std > 0:
                    normalized = (image_float - mean) / std
                    # Clip to reasonable range
                    normalized = np.clip(normalized, -3, 3)
                    # Scale to [0, 1]
                    normalized = (normalized + 3) / 6
                else:
                    normalized = image_float / 255.0
                    
            elif method == 'robust':
                # Robust normalization using percentiles
                p2, p98 = np.percentile(image_float, [2, 98])
                if p98 > p2:
                    normalized = np.clip((image_float - p2) / (p98 - p2), 0, 1)
                else:
                    normalized = image_float / 255.0
                    
            else:
                # Default: simple division by 255
                normalized = image_float / 255.0
            
            return np.clip(normalized, 0, 1)
            
        except Exception as e:
            logger.error(f"Error normalizing image: {e}")
            return None
    
    def apply_clahe(self, image: np.ndarray, clip_limit: float = None, 
                   grid_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
        
        Args:
            image: Input image array
            clip_limit: Threshold for contrast limiting
            grid_size: Size of the neighborhood for histogram equalization
            
        Returns:
            CLAHE processed image
        """
        if image is None:
            return None
        
        try:
            clip_limit = clip_limit or CLAHE_CLIP_LIMIT
            grid_size = grid_size or CLAHE_GRID_SIZE
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = image.copy()
            
            # Ensure image is in correct format for CLAHE
            if gray_image.dtype != np.uint8:
                # Convert to uint8
                gray_image = (gray_image * 255).astype(np.uint8)
            
            # Create CLAHE object
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            
            # Apply CLAHE
            clahe_image = clahe.apply(gray_image)
            
            # Convert back to original format if needed
            if len(image.shape) == 3:
                clahe_image = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2RGB)
            
            return clahe_image
            
        except Exception as e:
            logger.error(f"Error applying CLAHE: {e}")
            return None
    
    def enhance_contrast(self, image: np.ndarray, factor: float = 1.2) -> np.ndarray:
        """
        Enhance image contrast.
        
        Args:
            image: Input image array
            factor: Contrast enhancement factor (>1 increases contrast)
            
        Returns:
            Contrast enhanced image
        """
        if image is None:
            return None
        
        try:
            # Convert to PIL for enhancement
            if len(image.shape) == 3:
                pil_img = Image.fromarray((image * 255).astype(np.uint8))
            else:
                pil_img = Image.fromarray((image * 255).astype(np.uint8), mode='L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(pil_img)
            enhanced_img = enhancer.enhance(factor)
            
            # Convert back to numpy array
            enhanced_array = np.array(enhanced_img) / 255.0
            
            return enhanced_array
            
        except Exception as e:
            logger.error(f"Error enhancing contrast: {e}")
            return None
    
    def enhance_brightness(self, image: np.ndarray, factor: float = 1.1) -> np.ndarray:
        """
        Enhance image brightness.
        
        Args:
            image: Input image array
            factor: Brightness enhancement factor (>1 increases brightness)
            
        Returns:
            Brightness enhanced image
        """
        if image is None:
            return None
        
        try:
            # Convert to PIL for enhancement
            if len(image.shape) == 3:
                pil_img = Image.fromarray((image * 255).astype(np.uint8))
            else:
                pil_img = Image.fromarray((image * 255).astype(np.uint8), mode='L')
            
            # Enhance brightness
            enhancer = ImageEnhance.Brightness(pil_img)
            enhanced_img = enhancer.enhance(factor)
            
            # Convert back to numpy array
            enhanced_array = np.array(enhanced_img) / 255.0
            
            return enhanced_array
            
        except Exception as e:
            logger.error(f"Error enhancing brightness: {e}")
            return None
    
    def add_gaussian_noise(self, image: np.ndarray, variance: float = 0.01) -> np.ndarray:
        """
        Add Gaussian noise to image for augmentation.
        
        Args:
            image: Input image array
            variance: Noise variance
            
        Returns:
            Image with added noise
        """
        if image is None:
            return None
        
        try:
            noise = np.random.normal(0, np.sqrt(variance), image.shape)
            noisy_image = image + noise
            
            # Clip to valid range
            noisy_image = np.clip(noisy_image, 0, 1)
            
            return noisy_image
            
        except Exception as e:
            logger.error(f"Error adding noise: {e}")
            return None
    
    def rotate_image(self, image: np.ndarray, angle: float, 
                    fill_color: Union[int, Tuple[int, ...]] = 0) -> np.ndarray:
        """
        Rotate image by given angle.
        
        Args:
            image: Input image array
            angle: Rotation angle in degrees (positive = counter-clockwise)
            fill_color: Color for empty areas after rotation
            
        Returns:
            Rotated image
        """
        if image is None:
            return None
        
        try:
            # Convert to PIL for high-quality rotation
            if len(image.shape) == 3:
                pil_img = Image.fromarray((image * 255).astype(np.uint8))
            else:
                pil_img = Image.fromarray((image * 255).astype(np.uint8), mode='L')
            
            # Rotate image
            rotated_img = pil_img.rotate(angle, fillcolor=fill_color, expand=True)
            
            # Convert back to numpy array
            rotated_array = np.array(rotated_img) / 255.0
            
            return rotated_array
            
        except Exception as e:
            logger.error(f"Error rotating image: {e}")
            return None
    
    def flip_image(self, image: np.ndarray, direction: str = 'horizontal') -> np.ndarray:
        """
        Flip image horizontally or vertically.
        
        Args:
            image: Input image array
            direction: 'horizontal' or 'vertical'
            
        Returns:
            Flipped image
        """
        if image is None:
            return None
        
        try:
            if direction == 'horizontal':
                return np.fliplr(image)
            elif direction == 'vertical':
                return np.flipud(image)
            else:
                logger.warning(f"Unknown flip direction: {direction}")
                return image
                
        except Exception as e:
            logger.error(f"Error flipping image: {e}")
            return None
    
    def calculate_image_statistics(self, image: np.ndarray) -> dict:
        """
        Calculate various image statistics.
        
        Args:
            image: Input image array
            
        Returns:
            Dictionary with image statistics
        """
        if image is None:
            return {}
        
        try:
            stats = {}
            
            # Convert to grayscale for analysis if needed
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray_image = (image * 255).astype(np.uint8)
            
            # Basic statistics
            stats['mean'] = float(np.mean(gray_image))
            stats['std'] = float(np.std(gray_image))
            stats['min'] = float(np.min(gray_image))
            stats['max'] = float(np.max(gray_image))
            stats['median'] = float(np.median(gray_image))
            
            # Shape information
            stats['height'] = image.shape[0]
            stats['width'] = image.shape[1]
            stats['channels'] = image.shape[2] if len(image.shape) == 3 else 1
            stats['total_pixels'] = image.shape[0] * image.shape[1]
            
            # Contrast measure
            stats['contrast'] = float(np.std(gray_image))
            
            # Histogram statistics
            hist, _ = np.histogram(gray_image, bins=256, range=(0, 255))
            stats['histogram_entropy'] = float(-np.sum(hist * np.log2(hist + 1e-10)))
            
            # Detect if image might be too dark or too bright
            stats['too_dark'] = stats['mean'] < MIN_BRIGHTNESS
            stats['too_bright'] = stats['mean'] > MAX_BRIGHTNESS
            stats['low_contrast'] = stats['contrast'] < MIN_CONTRAST_THRESHOLD * 255
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating image statistics: {e}")
            return {}
    
    def validate_image_quality(self, image: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Validate image quality based on various metrics.
        
        Args:
            image: Input image array
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        if image is None:
            return False, ['Image is None']
        
        issues = []
        
        try:
            stats = self.calculate_image_statistics(image)
            
            # Check dimensions
            if stats.get('width', 0) < MIN_IMAGE_SIZE[0]:
                issues.append(f"Width too small: {stats.get('width', 0)} < {MIN_IMAGE_SIZE[0]}")
            
            if stats.get('height', 0) < MIN_IMAGE_SIZE[1]:
                issues.append(f"Height too small: {stats.get('height', 0)} < {MIN_IMAGE_SIZE[1]}")
            
            # Check brightness
            if stats.get('too_dark', False):
                issues.append(f"Image too dark: mean={stats.get('mean', 0):.1f}")
            
            if stats.get('too_bright', False):
                issues.append(f"Image too bright: mean={stats.get('mean', 0):.1f}")
            
            # Check contrast
            if stats.get('low_contrast', False):
                issues.append(f"Low contrast: std={stats.get('contrast', 0):.1f}")
            
            # Check for completely black or white images
            if stats.get('std', 0) < 1:
                issues.append("Image has no variation (completely uniform)")
            
            is_valid = len(issues) == 0
            return is_valid, issues
            
        except Exception as e:
            logger.error(f"Error validating image quality: {e}")
            return False, [f"Validation error: {e}"]
    
    def save_image(self, image: np.ndarray, output_path: Union[str, Path], 
                  quality: int = 95) -> bool:
        """
        Save image to file.
        
        Args:
            image: Image array to save
            output_path: Output file path
            quality: JPEG quality (if saving as JPEG)
            
        Returns:
            True if successful, False otherwise
        """
        if image is None:
            return False
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Ensure image is in correct format
            if image.dtype != np.uint8:
                # Convert from [0, 1] to [0, 255]
                image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            else:
                image_uint8 = image
            
            # Convert to PIL image
            if len(image_uint8.shape) == 3:
                pil_img = Image.fromarray(image_uint8)
            else:
                pil_img = Image.fromarray(image_uint8, mode='L')
            
            # Save with appropriate format
            if output_path.suffix.lower() in ['.jpg', '.jpeg']:
                pil_img.save(output_path, 'JPEG', quality=quality)
            elif output_path.suffix.lower() == '.png':
                pil_img.save(output_path, 'PNG')
            else:
                pil_img.save(output_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving image to {output_path}: {e}")
            return False
    
    def create_image_grid(self, images: List[np.ndarray], titles: List[str] = None, 
                         cols: int = 4, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create a grid of images for visualization.
        
        Args:
            images: List of image arrays
            titles: List of titles for each image
            cols: Number of columns in grid
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        try:
            n_images = len(images)
            rows = (n_images + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=figsize)
            
            # Handle single row case
            if rows == 1:
                axes = axes.reshape(1, -1)
            elif cols == 1:
                axes = axes.reshape(-1, 1)
            
            for i in range(n_images):
                row = i // cols
                col = i % cols
                ax = axes[row, col]
                
                # Display image
                if len(images[i].shape) == 3:
                    ax.imshow(images[i])
                else:
                    ax.imshow(images[i], cmap='gray')
                
                # Set title
                if titles and i < len(titles):
                    ax.set_title(titles[i])
                
                ax.axis('off')
            
            # Hide empty subplots
            for i in range(n_images, rows * cols):
                row = i // cols
                col = i % cols
                axes[row, col].axis('off')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating image grid: {e}")
            return None
    
    def preprocess_pipeline(self, image_path: Union[str, Path], 
                          target_size: Tuple[int, int] = None) -> Tuple[np.ndarray, dict]:
        """
        Complete preprocessing pipeline for a single image.
        
        Args:
            image_path: Path to input image
            target_size: Target size for resizing
            
        Returns:
            Tuple of (processed_image, processing_info)
        """
        target_size = target_size or IMAGE_SIZE
        processing_info = {
            'original_path': str(image_path),
            'steps_applied': [],
            'original_stats': {},
            'final_stats': {},
            'quality_issues': [],
            'success': False
        }
        
        try:
            # Step 1: Load image
            image = self.load_image(image_path, mode='RGB')
            if image is None:
                processing_info['error'] = 'Failed to load image'
                return None, processing_info
            
            # Record original statistics
            processing_info['original_stats'] = self.calculate_image_statistics(image)
            processing_info['steps_applied'].append('load')
            
            # Step 2: Validate quality
            is_valid, quality_issues = self.validate_image_quality(image)
            processing_info['quality_issues'] = quality_issues
            
            if not is_valid and not DEBUG_MODE:
                logger.warning(f"Image quality issues for {image_path}: {quality_issues}")
            
            # Step 3: Resize image
            if image.shape[:2] != target_size[::-1]:  # Note: PIL uses (width, height)
                image = self.resize_image(image, target_size, RESIZE_METHOD)
                processing_info['steps_applied'].append('resize')
            
            # Step 4: Apply CLAHE if enabled
            if APPLY_CLAHE:
                image = self.apply_clahe(image)
                processing_info['steps_applied'].append('clahe')
            
            # Step 5: Normalize pixels
            if NORMALIZE_PIXELS:
                image = self.normalize_pixels(image, method='minmax')
                processing_info['steps_applied'].append('normalize')
            
            # Record final statistics
            processing_info['final_stats'] = self.calculate_image_statistics(image)
            processing_info['success'] = True
            
            self.stats['processed_count'] += 1
            
            return image, processing_info
            
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline for {image_path}: {e}")
            processing_info['error'] = str(e)
            self.stats['error_count'] += 1
            return None, processing_info

def create_preprocessing_samples(image_paths: List[Union[str, Path]], 
                               output_dir: Union[str, Path],
                               num_samples: int = 10) -> None:
    """
    Create preprocessing samples for visualization.
    
    Args:
        image_paths: List of image paths
        output_dir: Output directory for samples
        num_samples: Number of samples to create
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processor = ImageProcessor()
    
    # Select random samples
    sample_paths = np.random.choice(image_paths, min(num_samples, len(image_paths)), replace=False)
    
    for i, image_path in enumerate(sample_paths):
        try:
            # Load original
            original = processor.load_image(image_path)
            if original is None:
                continue
            
            # Process image
            processed, info = processor.preprocess_pipeline(image_path)
            if processed is None:
                continue
            
            # Create comparison
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Original
            axes[0].imshow(original)
            axes[0].set_title(f'Original\nSize: {original.shape[:2]}')
            axes[0].axis('off')
            
            # Processed
            axes[1].imshow(processed)
            axes[1].set_title(f'Processed\nSize: {processed.shape[:2]}\nSteps: {len(info["steps_applied"])}')
            axes[1].axis('off')
            
            plt.suptitle(f'Sample {i+1}: {Path(image_path).name}')
            plt.tight_layout()
            
            # Save comparison
            output_file = output_dir / f'preprocessing_sample_{i+1}.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved preprocessing sample: {output_file}")
            
        except Exception as e:
            logger.error(f"Error creating sample for {image_path}: {e}")

def batch_preprocess_images(input_dir: Union[str, Path], 
                          output_dir: Union[str, Path],
                          target_size: Tuple[int, int] = None,
                          max_images: int = None) -> dict:
    """
    Batch preprocess images from input directory.
    
    Args:
        input_dir: Input directory containing images
        output_dir: Output directory for processed images
        target_size: Target size for all images
        max_images: Maximum number of images to process
        
    Returns:
        Processing statistics
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    target_size = target_size or IMAGE_SIZE
    processor = ImageProcessor()
    
    # Find all image files
    image_files = []
    for ext in SUPPORTED_FORMATS:
        image_files.extend(input_dir.rglob(f'*{ext}'))
        image_files.extend(input_dir.rglob(f'*{ext.upper()}'))
    
    if max_images:
        image_files = image_files[:max_images]
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process images
    processing_stats = {
        'total_found': len(image_files),
        'processed': 0,
        'failed': 0,
        'quality_issues': 0,
        'processing_times': []
    }
    
    from tqdm import tqdm
    import time
    
    for image_file in tqdm(image_files, desc="Processing images"):
        start_time = time.time()
        
        try:
            # Process image
            processed_img, info = processor.preprocess_pipeline(image_file, target_size)
            
            if processed_img is not None and info['success']:
                # Save processed image
                relative_path = image_file.relative_to(input_dir)
                output_path = output_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                success = processor.save_image(processed_img, output_path)
                
                if success:
                    processing_stats['processed'] += 1
                    if info['quality_issues']:
                        processing_stats['quality_issues'] += 1
                else:
                    processing_stats['failed'] += 1
            else:
                processing_stats['failed'] += 1
                logger.warning(f"Failed to process: {image_file}")
                
        except Exception as e:
            logger.error(f"Error processing {image_file}: {e}")
            processing_stats['failed'] += 1
        
        processing_time = time.time() - start_time
        processing_stats['processing_times'].append(processing_time)
    
    # Calculate final statistics
    processing_stats['avg_processing_time'] = np.mean(processing_stats['processing_times'])
    processing_stats['total_time'] = sum(processing_stats['processing_times'])
    processing_stats['success_rate'] = processing_stats['processed'] / processing_stats['total_found']
    
    logger.info(f"Batch processing complete:")
    logger.info(f"  - Processed: {processing_stats['processed']}/{processing_stats['total_found']}")
    logger.info(f"  - Success rate: {processing_stats['success_rate']:.2%}")
    logger.info(f"  - Avg time per image: {processing_stats['avg_processing_time']:.2f}s")
    logger.info(f"  - Images with quality issues: {processing_stats['quality_issues']}")
    
    return processing_stats

def main():
    """Main function for testing image utilities."""
    logger.info("Testing image processing utilities...")
    
    # Test with a sample image if available
    processor = ImageProcessor()
    
    # Print configuration
    print(f"Image Processing Configuration:")
    print(f"  Target size: {IMAGE_SIZE}")
    print(f"  Normalize pixels: {NORMALIZE_PIXELS}")
    print(f"  Apply CLAHE: {APPLY_CLAHE}")
    print(f"  Supported formats: {SUPPORTED_FORMATS}")
    
    logger.info("Image utilities ready for use!")

if __name__ == "__main__":
    main()