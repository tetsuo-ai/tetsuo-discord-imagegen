from PIL import Image
import subprocess
import tempfile
import logging
import traceback
import os
import gc
import shutil
import numpy as np
from typing import Optional, Dict, Any, List, Union, Tuple
from io import BytesIO
from pathlib import Path
from tetimi import ImageProcessor, EFFECT_ORDER, ANIMATION_PRESETS
from dataclasses import dataclass

@dataclass
class Keyframe:
    time: float  # 0.0 to 1.0
    value: Any
    easing: str = 'linear'  # Default to linear interpolation

def ease_value(start: float, end: float, progress: float, easing: str = 'linear') -> float:
    """Apply easing function to progress"""
    if easing == 'linear':
        return progress
    elif easing == 'ease_in':
        return progress * progress
    elif easing == 'ease_out':
        return -(progress - 1) ** 2 + 1
    elif easing == 'ease_in_out':
        if progress < 0.5:
            return 2 * progress * progress
        else:
            return 1 - (-2 * progress + 2) ** 2 / 2
    return progress

def interpolate_value(start_val: Any, end_val: Any, progress: float) -> Any:
    """Interpolate between two values based on progress"""
    if isinstance(start_val, (int, float)) and isinstance(end_val, (int, float)):
        return start_val + (end_val - start_val) * progress
    elif isinstance(start_val, tuple) and isinstance(end_val, tuple):
        return tuple(
            start + (end - start) * progress
            for start, end in zip(start_val, end_val)
        )
    return start_val

def get_keyframe_value(keyframes: List[Keyframe], time: float) -> Any:
    """Get interpolated value at given time from keyframes"""
    if not keyframes:
        return None
        
    # Handle boundary cases
    if time <= keyframes[0].time:
        return keyframes[0].value
    if time >= keyframes[-1].time:
        return keyframes[-1].value
        
    # Find surrounding keyframes
    for i in range(len(keyframes) - 1):
        k1, k2 = keyframes[i], keyframes[i + 1]
        if k1.time <= time <= k2.time:
            # Calculate progress between keyframes
            segment_progress = (time - k1.time) / (k2.time - k1.time)
            # Apply easing
            eased_progress = ease_value(0, 1, segment_progress, k2.easing)
            # Interpolate value
            return interpolate_value(k1.value, k2.value, eased_progress)
    
    return keyframes[-1].value

class AnimationProcessor:
    def __init__(self, image_input: Union[str, bytes, Image.Image, BytesIO], output_dir: str = "animations"):
        """Initialize animation processor with dimension handling"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('AnimationProcessor')
        
        # Handle different input types
        if isinstance(image_input, str):
            self.input_path = Path(image_input)
            if not self.input_path.exists():
                raise ValueError(f"Input image not found: {image_input}")
            image = Image.open(self.input_path)
            self.temp_input = False
        else:
            # For non-path inputs, save to temporary file
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            self.input_path = Path(tmp.name)
            self.temp_input = True
            
            if isinstance(image_input, bytes):
                image = Image.open(BytesIO(image_input))
            elif isinstance(image_input, Image.Image):
                image = image_input
            elif isinstance(image_input, BytesIO):
                image = Image.open(image_input)
            else:
                raise ValueError("Unsupported image input type")
            
            image.save(tmp.name, format='PNG')

        # Ensure dimensions are even for video encoding
        width, height = image.size
        new_width = width if width % 2 == 0 else width + 1
        new_height = height if height % 2 == 0 else height + 1
        
        if new_width != width or new_height != height:
            new_image = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 0))
            new_image.paste(image, ((new_width - width) // 2, (new_height - height) // 2))
            image = new_image
            
        self.base_image = image
        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / "frames"
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        if self.frames_dir.exists():
            shutil.rmtree(self.frames_dir)
        self.frames_dir.mkdir(exist_ok=True)
        
        # Initialize base ImageProcessor
        self.base_processor = ImageProcessor(self.base_image)

    def generate_frames(self, params: Dict[str, Any], num_frames: int = 60) -> List[Path]:
            """Generate animation frames with proper parameter interpolation"""
            frame_paths = []

            try:
                for i in range(num_frames):
                    # Calculate progress through animation (0.0 to 1.0)
                    progress = i / (num_frames - 1)
                    
                    # Calculate frame parameters
                    frame_params = {}
                    for effect, value in params.items():
                        if isinstance(value, tuple):
                            if isinstance(value[0], (int, float)):
                                # Interpolate between start and end values
                                start, end = value
                                frame_value = start + (end - start) * progress
                                frame_params[effect] = frame_value
                            elif isinstance(value[0], tuple):  # RGB values
                                # Interpolate each RGB component
                                start, end = value
                                frame_value = tuple(
                                    int(s + (e - s) * progress)
                                    for s, e in zip(start, end)
                                )
                                frame_params[effect] = frame_value
                        else:
                            frame_params[effect] = value

                    # Process frame
                    result = self.base_processor.base_image.copy()
                    for effect in EFFECT_ORDER:
                        if effect in frame_params:
                            processor = ImageProcessor(result)
                            result = processor.apply_effect(effect, frame_params)

                    # Save frame
                    frame_path = self.frames_dir / f"frame_{i:04d}.png"
                    result.save(frame_path)
                    frame_paths.append(frame_path)
                    
                    self.logger.info(f"Generated frame {i + 1}/{num_frames}")

            except Exception as e:
                self.logger.error(f"Frame generation error: {traceback.format_exc()}")
                raise

            return frame_paths

    def create_video(self, frame_rate: int = 24, output_name: str = "animation.mp4") -> Optional[Path]:
        """Create video with proper encoding settings"""
        self.logger.info(f"Creating video with frame rate: {frame_rate}")
        output_path = self.output_dir / output_name
        
        frame_files = list(self.frames_dir.glob("frame_*.png"))
        if not frame_files:
            self.logger.error("No frames found for video creation")
            raise ValueError("No frames found for video creation")
        
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-framerate', str(frame_rate),
            '-i', str(self.frames_dir / 'frame_%04d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            '-preset', 'medium',
            '-movflags', '+faststart',
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',  # Ensure dimensions are even
            str(output_path)
        ]
        
        try:
            result = subprocess.run(
                ffmpeg_cmd,
                check=True,
                capture_output=True,
                text=True
            )
            return output_path
        except subprocess.CalledProcessError as e:
            self.logger.error(f"ffmpeg error: {e.stderr}")
            return None
        finally:
            if hasattr(self, 'temp_input') and self.temp_input:
                try:
                    os.unlink(self.input_path)
                except Exception as e:
                    self.logger.error(f"Error cleaning up temp file: {e}")

    def cleanup(self):
        """Remove temporary files"""
        if self.frames_dir.exists():
            shutil.rmtree(self.frames_dir)
            
    def __del__(self):
        """Cleanup on object destruction"""
        self.cleanup()