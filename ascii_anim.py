from PIL import Image
import time
import os
from typing import List, Union, Optional
from io import BytesIO
import tempfile
import shutil
from pathlib import Path
import logging
from tetimi import ImageProcessor, EFFECT_PRESETS, EFFECT_ORDER

class ASCIIAnimationProcessor:
    def __init__(self, image_input: Union[str, bytes, Image.Image, BytesIO], output_dir: str = "animations"):
        """Initialize ASCII animation processor 
        
        Args:
            image_input: Input image in various formats (path, bytes, PIL Image, or BytesIO)
            output_dir: Directory for output files
        """
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger('ASCIIAnimationProcessor')
        
        # Handle different input types
        if isinstance(image_input, str):
            self.input_path = Path(image_input)
            if not self.input_path.exists():
                raise ValueError(f"Input image not found: {image_input}")
            self.base_processor = ImageProcessor(Image.open(self.input_path))
        else:
            # For non-path inputs, save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                if isinstance(image_input, bytes):
                    tmp.write(image_input)
                    self.base_processor = ImageProcessor(Image.open(BytesIO(image_input)))
                elif isinstance(image_input, Image.Image):
                    image_input.save(tmp.name, format='PNG')
                    self.base_processor = ImageProcessor(image_input)
                elif isinstance(image_input, BytesIO):
                    tmp.write(image_input.getvalue())
                    self.base_processor = ImageProcessor(Image.open(image_input))
                self.input_path = Path(tmp.name)
                self.temp_input = True

        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / "ascii_frames"
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.frames_dir.mkdir(exist_ok=True)

    def generate_ascii_frames(self, params: dict, num_frames: int = 60,
                              cols: int = 80, scale: float = 0.43) -> List[List[str]]:
        """Generate frames with animated parameter transitions"""
        ascii_frames = []
        
        for i in range(num_frames):
            progress = i / (num_frames - 1)
            frame_params = {}
            
            # Interpolate parameters
            for effect, value in params.items():
                if isinstance(value, tuple) and len(value) == 2:
                    start_val, end_val = value
                    frame_params[effect] = start_val + (end_val - start_val) * progress
                else:
                    frame_params[effect] = value
            
            # Initialize a fresh copy of the base image for this frame
            result = self.base_processor.base_image.copy()  # Assuming PIL Image
            
            # Apply effects using the ImageProcessor
            for effect in EFFECT_ORDER:
                if effect in frame_params:
                    print(f"Applying effect: {effect} with value: {frame_params[effect]}")
                    result = self.base_processor.apply_effect(effect, {effect: frame_params[effect]})
                    print(f"Effect applied: {effect}")
            
            # Convert to ASCII
            ascii_frame = self.base_processor.convertImageToAscii(result, scale=scale, cols=cols)
            ascii_frames.append(ascii_frame)
        
        return ascii_frames  # Only return ASCII frames to reduce memory usage

    def save_frames(self, frames: List[List[str]], output_file: str = "ascii_animation.txt") -> Path:
        """Save ASCII frames to file with frame markers"""
        output_path = self.output_dir / output_file
        
        # Ensure the output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for i, frame in enumerate(frames):
                f.write(f"=== Frame {i} ===\n")
                f.write('\n'.join(frame))
                f.write('\n\n')
                
        return output_path

    def cleanup(self):
        """Remove temporary files"""
        if hasattr(self, 'temp_input') and getattr(self, 'temp_input', False):
            try:
                os.unlink(self.input_path)
            except Exception as e:
                self.logger.error(f"Error cleaning up temp file: {e}")
                
        if self.frames_dir.exists():
            shutil.rmtree(self.frames_dir)

    def __del__(self):
        """Cleanup on object destruction"""
        self.cleanup()
