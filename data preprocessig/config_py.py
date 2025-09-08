"""
Configuration file for bone fracture detection preprocessing pipeline.
Contains all constants, paths, and hyperparameters used across the project.
"""

import os
from pathlib import Path

# ======================== PROJECT PATHS ========================
# Base project directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data directories
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_ROOT / "raw" / "train"
PROCESSED_DATA_PATH = DATA_ROOT / "processed"
AUGMENTED_DATA_PATH = DATA_ROOT / "augmented"
METADATA_PATH = DATA_ROOT / "metadata"
REPORTS_PATH = DATA_ROOT / "reports"

# Create directories if they don't exist
for path in [DATA_ROOT, PROCESSED_DATA_PATH, AUGMENTED_DATA_PATH, METADATA_PATH, REPORTS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# ======================== IMAGE PROCESSING ========================
# Image dimensions
IMAGE_SIZE = (224, 224)  # Standard size for most pre-trained models
IMAGE_CHANNELS = 3  # RGB channels
RESIZE_METHOD = 'bilinear'  # 'bilinear', 'nearest', 'bicubic'

# Image format settings
SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.dicom']
OUTPUT_FORMAT = 'PNG'
IMAGE_QUALITY = 95  # For JPEG compression

# Preprocessing parameters
NORMALIZE_PIXELS = True  # Scale pixels to [0, 1]
APPLY_CLAHE = True  # Contrast Limited Adaptive Histogram Equalization
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)

# ======================== DATA SPLITTING ========================
# Train/Validation/Test split ratios (must sum to 1.0)
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.1

# Random seed for reproducible splits
RANDOM_SEED = 42
SHUFFLE_DATA = True

# Patient-level split (recommended for medical data)
PATIENT_LEVEL_SPLIT = True  # If True, ensures same patient doesn't appear in multiple splits

# ======================== DATA AUGMENTATION ========================
# Augmentation parameters
AUGMENTATION_ENABLED = True
AUGMENT_TRAINING_ONLY = True  # Only augment training data

# Rotation
ROTATION_RANGE = 15  # degrees
ROTATION_PROBABILITY = 0.5

# Brightness and contrast
BRIGHTNESS_RANGE = 0.2  # ±20% brightness variation
CONTRAST_RANGE = 0.2    # ±20% contrast variation
BRIGHTNESS_PROBABILITY = 0.3
CONTRAST_PROBABILITY = 0.3

# Geometric transformations
HORIZONTAL_FLIP = False  # Careful with medical images
VERTICAL_FLIP = False    # Usually not recommended for X-rays
ZOOM_RANGE = 0.1        # ±10% zoom
TRANSLATION_RANGE = 0.1  # ±10% translation
ZOOM_PROBABILITY = 0.3
TRANSLATION_PROBABILITY = 0.3

# Noise addition
ADD_NOISE = True
NOISE_VARIANCE = 0.01  # Gaussian noise variance
NOISE_PROBABILITY = 0.2

# ======================== BATCH PROCESSING ========================
# Batch sizes
BATCH_SIZE = 32
VALIDATION_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16

# Processing parameters
NUM_WORKERS = 4  # Number of parallel workers for data loading
PREFETCH_BUFFER = 2  # Number of batches to prefetch

# ======================== CLASS LABELS ========================
# Label mapping
CLASS_NAMES = ['negative', 'positive']  # Order matters for encoding
NUM_CLASSES = len(CLASS_NAMES)

# Label encoding
LABEL_ENCODING = {
    'negative': 0,
    'positive': 1
}

# Reverse mapping for decoding
LABEL_DECODING = {v: k for k, v in LABEL_ENCODING.items()}

# ======================== STUDY TYPES ========================
# Supported study types from your dataset
STUDY_TYPES = [
    'XR_ELBOW',
    'XR_FINGER',
    'XR_FOREARM',
    'XR_HAND',
    'XR_HUMERUS',
    'XR_SHOULDER',
    'XR_WRIST'
]

# Focus study type (set None to use all)
FOCUS_STUDY_TYPE = 'XR_ELBOW'  # Start with elbow, expand later

# ======================== QUALITY CONTROL ========================
# Image quality thresholds
MIN_IMAGE_SIZE = (64, 64)      # Minimum acceptable image size
MAX_IMAGE_SIZE = (2048, 2048)  # Maximum image size before resizing
MIN_FILE_SIZE = 1024           # Minimum file size in bytes (1KB)
MAX_FILE_SIZE = 50 * 1024 * 1024  # Maximum file size in bytes (50MB)

# Quality metrics thresholds
MIN_CONTRAST_THRESHOLD = 0.1   # Minimum contrast ratio
MAX_BLUR_THRESHOLD = 100       # Maximum blur (lower = more blurred)
MIN_BRIGHTNESS = 10            # Minimum average brightness
MAX_BRIGHTNESS = 245           # Maximum average brightness

# ======================== LOGGING AND REPORTING ========================
# Logging configuration
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
SAVE_PROCESSING_LOGS = True
LOG_FILE = REPORTS_PATH / 'preprocessing.log'

# Report generation
GENERATE_HTML_REPORTS = True
GENERATE_PDF_REPORTS = False
SAVE_SAMPLE_IMAGES = True
NUM_SAMPLE_IMAGES = 20

# Progress tracking
SHOW_PROGRESS_BAR = True
PROGRESS_UPDATE_FREQUENCY = 100  # Update every N images

# ======================== PERFORMANCE SETTINGS ========================
# Memory management
MAX_MEMORY_USAGE = 0.8  # Maximum RAM usage (80%)
ENABLE_MEMORY_MAPPING = True
CACHE_PREPROCESSED_IMAGES = False  # Cache on disk

# Multiprocessing
USE_MULTIPROCESSING = True
MAX_PROCESSES = None  # None = use all available cores

# ======================== VALIDATION SETTINGS ========================
# Cross-validation
ENABLE_CROSS_VALIDATION = False
CV_FOLDS = 5

# Stratification
STRATIFY_SPLITS = True  # Ensure balanced class distribution

# ======================== EXPORT SETTINGS ========================
# Output file formats
EXPORT_CSV = True
EXPORT_JSON = True
EXPORT_PICKLE = False

# Metadata to save
SAVE_IMAGE_METADATA = True
SAVE_PROCESSING_STATISTICS = True
SAVE_AUGMENTATION_PARAMETERS = True

# ======================== DEBUGGING ========================
# Debug settings
DEBUG_MODE = False
SAVE_INTERMEDIATE_RESULTS = DEBUG_MODE
VERBOSE_LOGGING = DEBUG_MODE

# Sample size for debugging
DEBUG_SAMPLE_SIZE = 100  # Use smaller dataset for testing

# ======================== HELPER FUNCTIONS ========================
def get_processed_path(split_type):
    """Get path for processed data split."""
    return PROCESSED_DATA_PATH / split_type

def get_augmented_path():
    """Get path for augmented training data."""
    return AUGMENTED_DATA_PATH / 'train'

def get_metadata_file(filename):
    """Get full path for metadata file."""
    return METADATA_PATH / filename

def get_report_file(filename):
    """Get full path for report file."""
    return REPORTS_PATH / filename

def validate_config():
    """Validate configuration settings."""
    # Check split ratios
    total_ratio = TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    
    # Check image size
    if len(IMAGE_SIZE) != 2:
        raise ValueError("IMAGE_SIZE must be a tuple of (width, height)")
    
    # Check batch sizes
    if BATCH_SIZE <= 0 or VALIDATION_BATCH_SIZE <= 0 or TEST_BATCH_SIZE <= 0:
        raise ValueError("All batch sizes must be positive integers")
    
    # Check class names
    if len(CLASS_NAMES) != NUM_CLASSES:
        raise ValueError("NUM_CLASSES must match length of CLASS_NAMES")
    
    return True

# Validate configuration on import
if __name__ == "__main__":
    validate_config()
    print("Configuration validated successfully!")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data root: {DATA_ROOT}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Classes: {CLASS_NAMES}")
else:
    validate_config()