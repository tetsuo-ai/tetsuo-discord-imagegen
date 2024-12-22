from PIL import Image
import numpy as np
import subprocess
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from io import BytesIO
import os

from .image_processor import BaseImageProcessor
from .effect_processor import EffectProcessor
from .utils import ImageUtils

class AnimationProcessor:
    """
    Handles creation and management of image effect animations with improved multi-effect support.
    """
    
    def __init__(self, image_input: Union[str, bytes, Image.Image, BytesIO]):
        """Initialize animation processor with enhanced effect handling."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('AnimationProcessor')
        
        # Load and validate input image
        self.base_image = ImageUtils.load_image(image_input)
        
        # Ensure dimensions are even for video encoding
        width, height = self.base_image.size
        new_width, new_height = ImageUtils.ensure_size_even(width, height)
        
        if (new_width, new_height) != (width, height):
            new_image = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 0))
            new_image.paste(self.base_image, ((new_width - width) // 2, 
                                            (new_height - height) // 2))
            self.base_image = new_image
        
        # Set up temporary directory for frames
        self.temp_dir = Path(tempfile.mkdtemp(prefix='anim_frames_'))
        self.frames_dir = self.temp_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)
        
    def _validate_effects(self, effects: List[Tuple[str, Dict[str, Any]]]) -> None:
        """Validate effect parameters before processing."""
        valid_effects = {
            'glitch', 'chroma', 'scan', 'noise', 'energy', 'pulse', 'consciousness'
        }
        
        for effect_name, params in effects:
            if effect_name not in valid_effects:
                raise ValueError(f"Invalid effect: {effect_name}")
                
            # Normalize intensity parameters
            if 'intensity' in params:
                intensity = params['intensity']
                if isinstance(intensity, (int, float)):
                    params['intensity'] = min(1.0, max(0.0, float(intensity) / 100))
                elif isinstance(intensity, tuple):
                    params['intensity'] = (
                        min(1.0, max(0.0, float(intensity[0]) / 100)),
                        min(1.0, max(0.0, float(intensity[1]) / 100))
                    )

    def _interpolate_parameters(self, params: Dict[str, Any], progress: float) -> Dict[str, Any]:
        """Interpolate effect parameters for current frame."""
        frame_params = {}
        for param_name, param_value in params.items():
            if isinstance(param_value, tuple) and len(param_value) == 2:
                frame_params[param_name] = ImageUtils.interpolate_value(
                    param_value[0], param_value[1], progress
                )
            else:
                frame_params[param_name] = param_value
        return frame_params

    def generate_frames(self, 
                       effects: List[Tuple[str, Dict[str, Any]]], 
                       num_frames: int = 30) -> List[Path]:
        """Generate animation frames with improved multi-effect support."""
        frame_paths = []
        self._validate_effects(effects)
        
        try:
            for i in range(num_frames):
                progress = i / (num_frames - 1)
                
                # Start with fresh copy of base image for each frame
                frame = self.base_image.copy()
                processor = EffectProcessor(frame)
                
                # Apply effects in sequence with proper parameter interpolation
                for effect_name, params in effects:
                    frame_params = self._interpolate_parameters(params, progress)
                    
                    try:
                        processor.apply_effect(effect_name, frame_params)
                    except Exception as e:
                        self.logger.warning(f"Error applying effect {effect_name}: {str(e)}")
                        continue
                
                # Save frame
                frame_path = self.frames_dir / f"frame_{i:04d}.png"
                processor.save(frame_path)
                frame_paths.append(frame_path)
                
                self.logger.info(f"Generated frame {i + 1}/{num_frames}")
                
        except Exception as e:
            self.logger.error(f"Frame generation error: {str(e)}")
            raise
            
        return frame_paths

    def create_video(self, 
                    frame_paths: List[Path],
                    output_path: Optional[Union[str, Path]] = None,
                    frame_rate: int = 24,
                    crf: int = 23,
                    preset: str = 'medium') -> Optional[Path]:
        """Create video with improved error handling and frame management."""
        if not frame_paths:
            raise ValueError("No frames provided for video creation")
            
        if not output_path:
            output_path = self.temp_dir / "output.mp4"
        else:
            output_path = Path(output_path)
            
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Verify all frames exist
            for frame_path in frame_paths:
                if not frame_path.exists():
                    raise FileNotFoundError(f"Missing frame: {frame_path}")
            
            # Construct ffmpeg command
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-framerate', str(frame_rate),
                '-i', str(self.frames_dir / 'frame_%04d.png'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', str(crf),
                '-preset', preset,
                '-movflags', '+faststart',
                '-vf', 'format=yuv420p',
                str(output_path)
            ]
            
            # Run ffmpeg with proper error handling
            try:
                result = subprocess.run(
                    ffmpeg_cmd,
                    check=True,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    return output_path
                    
            except subprocess.CalledProcessError as e:
                self.logger.error(f"ffmpeg error: {e.stderr}")
                raise
                
        except Exception as e:
            self.logger.error(f"Video creation error: {str(e)}")
            raise
            
        return None

    def cleanup(self):
        """Clean up temporary files with improved error handling."""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")

    def __del__(self):
        """Ensure cleanup on object destruction."""
        self.cleanup()