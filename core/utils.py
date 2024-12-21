from PIL import Image, ImageStat
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union, List, TypeVar, Callable
from pathlib import Path
import math
from io import BytesIO

# Type definitions
Number = TypeVar('Number', int, float)
ImageType = Union[str, bytes, Image.Image, BytesIO, Path]
ParamValue = Union[Number, Tuple[Number, Number], List[Number]]

class ImageUtils:
    """Utility functions for image processing operations."""
    
    @staticmethod
    def load_image(image_input: ImageType) -> Image.Image:
        """
        Load an image from various input types.
        
        Args:
            image_input: Can be:
                - str/Path: Path to image file
                - bytes: Raw image data
                - Image.Image: PIL Image object
                - BytesIO: BytesIO containing image data
        
        Returns:
            PIL.Image: Loaded image
            
        Raises:
            ValueError: If input type is unsupported or file not found
            IOError: If image file cannot be opened
        """
        try:
            if isinstance(image_input, (str, Path)):
                path = Path(image_input)
                if not path.exists():
                    raise ValueError(f"Input image not found: {image_input}")
                return Image.open(path)
            elif isinstance(image_input, bytes):
                return Image.open(BytesIO(image_input))
            elif isinstance(image_input, Image.Image):
                return image_input
            elif isinstance(image_input, BytesIO):
                return Image.open(image_input)
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
        except Exception as e:
            raise IOError(f"Failed to load image: {str(e)}")

    @staticmethod
    def ensure_rgb(image: Image.Image) -> Image.Image:
        """
        Ensure image is in RGB mode.
        
        Args:
            image: Input image
            
        Returns:
            Image in RGB mode
        """
        if image.mode != 'RGB':
            return image.convert('RGB')
        return image

    @staticmethod
    def ensure_rgba(image: Image.Image) -> Image.Image:
        """
        Ensure image is in RGBA mode.
        
        Args:
            image: Input image
            
        Returns:
            Image in RGBA mode
        """
        if image.mode != 'RGBA':
            return image.convert('RGBA')
        return image

    @staticmethod
    def get_image_stats(image: Image.Image) -> Dict[str, float]:
        """
        Calculate comprehensive image statistics.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary containing image statistics
        """
        analysis_image = ImageUtils.ensure_rgb(image)
        stat = ImageStat.Stat(analysis_image)
        
        # Basic statistics
        brightness = sum(stat.mean) / (3 * 255.0)
        contrast = sum(stat.stddev) / (3 * 255.0)
        
        # Color analysis
        r, g, b = stat.mean
        color_variance = np.std([r, g, b])
        
        # Calculate color distribution
        hist = analysis_image.histogram()
        total_pixels = sum(hist)
        color_distribution = [sum(hist[i:i+256]) / total_pixels for i in range(0, 768, 256)]
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'color_variance': color_variance,
            'color_distribution': color_distribution,
            'mean_values': stat.mean,
            'stddev_values': stat.stddev
        }

    @staticmethod
    def interpolate_value(start: ParamValue, 
                         end: ParamValue, 
                         progress: float) -> ParamValue:
        """
        Interpolate between two values or tuples of values.
        
        Args:
            start: Starting value(s)
            end: Ending value(s)
            progress: Interpolation progress (0.0 to 1.0)
            
        Returns:
            Interpolated value(s)
            
        Raises:
            ValueError: If values are incompatible types
        """
        if not 0 <= progress <= 1:
            raise ValueError("Progress must be between 0 and 1")
            
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            return start + (end - start) * progress
            
        if isinstance(start, (tuple, list)) and isinstance(end, (tuple, list)):
            if len(start) != len(end):
                raise ValueError("Start and end sequences must have same length")
            return type(start)(
                start[i] + (end[i] - start[i]) * progress 
                for i in range(len(start))
            )
            
        raise ValueError(f"Cannot interpolate between {type(start)} and {type(end)}")

    @staticmethod
    def apply_easing(progress: float, easing_type: str = 'linear') -> float:
        """
        Apply easing function to progress value.
        
        Args:
            progress: Raw progress value (0.0 to 1.0)
            easing_type: Type of easing to apply
                Options: 'linear', 'ease_in', 'ease_out', 'ease_in_out'
                
        Returns:
            Eased progress value
            
        Raises:
            ValueError: If progress is out of range or easing type unknown
        """
        if not 0 <= progress <= 1:
            raise ValueError("Progress must be between 0 and 1")
            
        if easing_type == 'linear':
            return progress
        elif easing_type == 'ease_in':
            return progress * progress
        elif easing_type == 'ease_out':
            return 1 - (1 - progress) * (1 - progress)
        elif easing_type == 'ease_in_out':
            if progress < 0.5:
                return 2 * progress * progress
            else:
                return 1 - (-2 * progress + 2) ** 2 / 2
        else:
            raise ValueError(f"Unknown easing type: {easing_type}")

    @staticmethod
    def validate_params(params: Dict[str, Any], 
                       validators: Dict[str, Callable[[Any], bool]],
                       required: Optional[List[str]] = None) -> None:
        """
        Validate parameter dictionary against constraints.
        
        Args:
            params: Parameter dictionary to validate
            validators: Dictionary of validation functions
            required: List of required parameter names
            
        Raises:
            ValueError: If validation fails
        """
        if required:
            missing = [param for param in required if param not in params]
            if missing:
                raise ValueError(f"Missing required parameters: {', '.join(missing)}")
        
        for param_name, validator in validators.items():
            if param_name in params:
                value = params[param_name]
                if not validator(value):
                    raise ValueError(f"Invalid value for {param_name}: {value}")

    @staticmethod
    def ensure_size_even(width: int, height: int) -> Tuple[int, int]:
        """
        Ensure dimensions are even (required for some video encoders).
        
        Args:
            width: Original width
            height: Original height
            
        Returns:
            Tuple of adjusted width and height
        """
        return (
            width if width % 2 == 0 else width + 1,
            height if height % 2 == 0 else height + 1
        )

    @staticmethod
    def create_temp_path(prefix: str = "", 
                        suffix: str = "", 
                        directory: Optional[Union[str, Path]] = None) -> Path:
        """
        Create a temporary file path.
        
        Args:
            prefix: Filename prefix
            suffix: Filename suffix
            directory: Parent directory (optional)
            
        Returns:
            Path object for temporary file
        """
        import tempfile
        
        if directory:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
        
        with tempfile.NamedTemporaryFile(
            prefix=prefix,
            suffix=suffix,
            dir=directory,
            delete=False
        ) as tmp:
            return Path(tmp.name)

    @staticmethod
    def cleanup_temp_files(paths: List[Union[str, Path]]) -> None:
        """
        Safely delete temporary files.
        
        Args:
            paths: List of file paths to delete
        """
        for path in paths:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception as e:
                print(f"Warning: Failed to delete {path}: {e}")