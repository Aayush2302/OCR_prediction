class PreProcessingConfig:
    """Configuration for image preprocessing operations."""
    
    # ========================================
    # RESIZE - Critical for small text OCR!
    # ========================================
    TARGET_SIZE = 2400  # 🔥 DOUBLE your current 1200 - Critical!
    TARGET_HEIGHT = None
    
    # ========================================
    # DESKEWING - Your images look straight
    # ========================================
    ENABLE_DESKEW = False  # 🔥 Disable - saves time, your images are straight
    
    # ========================================
    # CLAHE - Contrast Enhancement
    # ========================================
    USE_CLAHE = True
    CLAHE_CLIP_LIMIT = 2.5  # 🔥 Slightly higher than your 2.0
    CLAHE_TILE_GRID_SIZE = 8
    
    # ========================================
    # DENOISING - CRITICAL for your noisy images
    # ========================================
    DENOISING_METHOD = 'bilateral'  # 🔥 Keep bilateral
    BILATERAL_DIAMETER = 5  # 🔥 Reduced from 9 - gentler
    BILATERAL_SIGMA_COLOR = 50  # 🔥 Reduced from 75
    BILATERAL_SIGMA_SPACE = 50  # 🔥 Reduced from 75
    
    # Alternative methods (not used but available)
    GAUSSIAN_KERNEL_SIZE = 5
    MEDIAN_KERNEL_SIZE = 3
    
    # ========================================
    # THRESHOLDING - Convert to binary
    # ========================================
    THRESHOLD_METHOD = 'adaptive'  # 🔥 Good choice!
    ADAPTIVE_BLOCK_SIZE = 15  # 🔥 Reduced from 21 - better for text
    ADAPTIVE_C = 5  # 🔥 Reduced from 10 - less aggressive
    
    # ========================================
    # MORPHOLOGY - Clean up binary image
    # ========================================
    MORPHOLOGY_METHOD = 'closing'  # 🔥 Connects broken text
    MORPHOLOGY_KERNEL_SIZE = 2  # 🔥 Reduced from 3 - preserve details
    
    # ========================================
    # ARTIFACT STORAGE
    # ========================================
    SAVE_INTERMEDIATE_STEPS = True

    SMART_DENOISE = True
    
    # ========================================
    # DEBUG
    # ========================================
    SHOW_STEPS = False