"""
Configuration with auto-orientation enabled.
"""

class PreProcessingConfig:
    """Configuration for image preprocessing operations."""
    
    # ========================================
    # AUTO-ORIENTATION (NEW!)
    # ========================================
    AUTO_ORIENT = True  # 🔥 Enable automatic orientation correction
    
    # ========================================
    # RESIZE
    # ========================================
    TARGET_SIZE = 1200  # Your current size (works for you)
    TARGET_HEIGHT = None
    
    # ========================================
    # DESKEWING
    # ========================================
    ENABLE_DESKEW = False  # You have this disabled
    
    # ========================================
    # CLAHE
    # ========================================
    USE_CLAHE = True
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_GRID_SIZE = 8
    
    # ========================================
    # THRESHOLDING
    # ========================================
    THRESHOLD_METHOD = "adaptive"
    ADAPTIVE_BLOCK_SIZE = 21
    ADAPTIVE_C = 10
    
    # ========================================
    # DENOISING (you have disabled)
    # ========================================
    DENOISING_METHOD = 'gaussian'
    GAUSSIAN_KERNEL_SIZE = 5
    MEDIAN_KERNEL_SIZE = 3
    BILATERAL_DIAMETER = 9
    BILATERAL_SIGMA_COLOR = 75
    BILATERAL_SIGMA_SPACE = 75
    
    # ========================================
    # MORPHOLOGY
    # ========================================
    MORPHOLOGY_METHOD = 'closing'
    MORPHOLOGY_KERNEL_SIZE = 3
    
    # ========================================
    # ARTIFACT STORAGE
    # ========================================
    SAVE_INTERMEDIATE_STEPS = True
    
    # ========================================
    # DEBUG
    # ========================================
    SHOW_STEPS = False
