from PIL import Image, ImageDraw, ImageFont
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
    Handles creation and management of image effect animations.
    """
    
    def __init__(self, image_input: Union[str, bytes, Image.Image, BytesIO]):
        """
        Initialize animation processor.
        
        Args:
            image_input: Source image in various formats
        """
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
        
        # Initialize processors
        self.effect_processor = EffectProcessor(self.base_image)
        
    def generate_frames(self, 
                       effects: List[Tuple[str, Dict[str, Any]]], 
                       num_frames: int = 30) -> List[Path]:
        """
        Generate animation frames with multiple effects.
        
        Args:
            effects: List of (effect_name, parameters) tuples
            num_frames: Number of frames to generate
            
        Returns:
            List of paths to generated frame files
        """
        frame_paths = []
        
        try:
            for i in range(num_frames):
                # Calculate animation progress
                progress = i / (num_frames - 1)
                
                # Process frame with interpolated parameters
                frame = self.base_image.copy()
                processor = EffectProcessor(frame)
                
                for effect_name, params in effects:
                    # Interpolate parameters
                    frame_params = {}
                    for param_name, param_value in params.items():
                        if isinstance(param_value, tuple) and len(param_value) == 2:
                            frame_params[param_name] = ImageUtils.interpolate_value(
                                param_value[0], param_value[1], progress
                            )
                        else:
                            frame_params[param_name] = param_value
                    
                    # Apply effect
                    processor.apply_effect(effect_name, frame_params)
                
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
        """
        Create video from frames using ffmpeg.
        
        Args:
            frame_paths: List of frame file paths
            output_path: Path for output video file
            frame_rate: Frames per second
            crf: Constant Rate Factor (18-28 recommended)
            preset: ffmpeg encoding preset
            
        Returns:
            Path to output video file
        """
        if not frame_paths:
            raise ValueError("No frames provided for video creation")
            
        if not output_path:
            output_path = self.temp_dir / "output.mp4"
        else:
            output_path = Path(output_path)
            
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
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
            
            # Run ffmpeg
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
        except Exception as e:
            self.logger.error(f"Video creation error: {str(e)}")
            
        return None

    def create_gif(self, 
                  frame_paths: List[Path],
                  output_path: Optional[Union[str, Path]] = None,
                  duration: int = 50) -> Optional[Path]:
        """
        Create animated GIF from frames.
        
        Args:
            frame_paths: List of frame file paths
            output_path: Path for output GIF file
            duration: Frame duration in milliseconds
            
        Returns:
            Path to output GIF file
        """
        if not frame_paths:
            raise ValueError("No frames provided for GIF creation")
            
        if not output_path:
            output_path = self.temp_dir / "output.gif"
        else:
            output_path = Path(output_path)
            
        try:
            # Load frames and optimize for GIF
            frames = []
            for frame_path in frame_paths:
                with Image.open(frame_path) as frame:
                    # Convert to P mode with adaptive palette
                    if frame.mode != 'P':
                        frame = frame.convert('RGBA').convert(
                            'P', 
                            palette=Image.Palette.ADAPTIVE, 
                            colors=256
                        )
                    frames.append(frame.copy())
            
            # Save as GIF
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0,
                optimize=True
            )
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"GIF creation error: {str(e)}")
            return None

    def create_ascii_animation(self,
                             effects: List[Tuple[str, Dict[str, Any]]],
                             num_frames: int = 30,
                             cols: int = 120,
                             scale: float = 0.43) -> List[str]:
        """
        Create ASCII art animation frames.
        
        Args:
            effects: List of (effect_name, parameters) tuples
            num_frames: Number of frames to generate
            cols: Number of columns for ASCII art
            scale: Character aspect ratio adjustment
            
        Returns:
            List of ASCII art strings
        """
        ascii_frames = []
        
        try:
            for i in range(num_frames):
                progress = i / (num_frames - 1)
                
                # Process frame with effects
                frame = self.base_image.copy()
                processor = EffectProcessor(frame)
                
                for effect_name, params in effects:
                    frame_params = {}
                    for param_name, param_value in params.items():
                        if isinstance(param_value, tuple) and len(param_value) == 2:
                            frame_params[param_name] = ImageUtils.interpolate_value(
                                param_value[0], param_value[1], progress
                            )
                        else:
                            frame_params[param_name] = param_value
                    
                    processor.apply_effect(effect_name, frame_params)
                
                # Convert to ASCII
                ascii_frame = processor.convertImageToAscii(cols=cols, scale=scale)
                ascii_frames.append('\n'.join(ascii_frame))
                
                self.logger.info(f"Generated ASCII frame {i + 1}/{num_frames}")
                
        except Exception as e:
            self.logger.error(f"ASCII animation error: {str(e)}")
            raise
            
        return ascii_frames

    def cleanup(self):
        """Clean up temporary files."""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")

    def __del__(self):
        """Ensure cleanup on object destruction."""
        self.cleanup()