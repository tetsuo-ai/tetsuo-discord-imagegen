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
from tetimi import ImageProcessor, EFFECT_ORDER


class AnimationProcessor:
    def __init__(self, image_input: Union[str, bytes, Image.Image, BytesIO], output_dir: str = "animations"):
        """Initialize animation processor with dimension handling
        
        Args:
            image_input: Input image in various formats (path, bytes, PIL Image, or BytesIO)
            output_dir: Directory for output files
        """
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger('AnimationProcessor')
        
        # Handle different input types
        if isinstance(image_input, str):
            self.input_path = Path(image_input)
            if not self.input_path.exists():
                raise ValueError(f"Input image not found: {image_input}")
            image = Image.open(self.input_path)
        else:
            # For non-path inputs, save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                if isinstance(image_input, bytes):
                    tmp.write(image_input)
                    image = Image.open(BytesIO(image_input))
                elif isinstance(image_input, Image.Image):
                    image_input.save(tmp.name, format='PNG')
                    image = image_input
                elif isinstance(image_input, BytesIO):
                    tmp.write(image_input.getvalue())
                    image = Image.open(image_input)
                self.input_path = Path(tmp.name)
                self.temp_input = True

        # Ensure dimensions are even for video encoding
        width, height = image.size
        new_width = width if width % 2 == 0 else width + 1
        new_height = height if height % 2 == 0 else height + 1
        
        if new_width != width or new_height != height:
            # Resize with padding to maintain aspect ratio
            new_image = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 0))
            new_image.paste(image, ((new_width - width) // 2, (new_height - height) // 2))
            image = new_image
            
        self.base_image = image
        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / "frames"
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.frames_dir.mkdir(exist_ok=True)
        
        # Initialize base ImageProcessor
        self.base_processor = ImageProcessor(self.base_image)

    def interpolate_params(self, start_value: Any, end_value: Any, progress: float) -> Any:
        """Interpolate between parameter values based on animation progress
        
        Args:
            start_value: Starting parameter value
            end_value: Ending parameter value
            progress: Animation progress (0.0 to 1.0)
            
        Returns:
            Interpolated value
        """
        if isinstance(start_value, tuple) and isinstance(end_value, tuple):
            # Handle color tuples
            if len(start_value) == 3 and len(end_value) == 3:
                return tuple(int(start + (end - start) * progress)
                           for start, end in zip(start_value, end_value))
        elif isinstance(start_value, (int, float)) and isinstance(end_value, (int, float)):
            # Handle numeric values
            return start_value + (end_value - start_value) * progress
        
        return start_value  # Default to start value if interpolation not possible

    def generate_frames(self, preset_name: Optional[str] = None,
                       params: Optional[Dict[str, Any]] = None,
                       num_frames: int = 60) -> List[Path]:
        """Generate animation frames with consistent dimensions
        
        Args:
            preset_name: Name of animation preset to use
            params: Custom animation parameters
            num_frames: Number of frames to generate
            
        Returns:
            List of frame file paths
        """
        frame_paths = []
        
        # Get parameters from preset or use custom params
        if preset_name and preset_name in ANIMATION_PRESETS:
            animation_params = ANIMATION_PRESETS[preset_name]
        elif params:
            animation_params = params
        else:
            raise ValueError("Either preset_name or params must be provided")
        
        for i in range(num_frames):
            progress = i / (num_frames - 1)
            
            try:
                # Calculate interpolated parameters for this frame
                frame_params = {}
                for effect, value in animation_params.items():
                    if isinstance(value, tuple) and len(value) == 2:
                        # Handle parameter ranges
                        start_val, end_val = value
                        frame_params[effect] = self.interpolate_params(start_val, end_val, progress)
                    else:
                        # Use static value
                        frame_params[effect] = value
                
                # Create frame
                result = self.base_processor.base_image
                
                # Apply effects in order
                for effect in EFFECT_ORDER:
                    if effect in frame_params:
                        self.base_processor.base_image = result
                        result = self.base_processor.apply_effect(effect, {effect: frame_params[effect]})
                
                # Ensure frame dimensions remain consistent
                if result.size != self.base_image.size:
                    new_frame = Image.new('RGBA', self.base_image.size, (0, 0, 0, 0))
                    new_frame.paste(result, ((self.base_image.size[0] - result.size[0]) // 2,
                                           (self.base_image.size[1] - result.size[1]) // 2))
                    result = new_frame
                
                # Save frame
                frame_path = self.frames_dir / f"frame_{i:04d}.png"
                result.save(frame_path)
                
                # Cleanup
                del result
                gc.collect()
                
                frame_paths.append(frame_path)
                
            except Exception as e:
                self.logger.error(f"Frame generation error: {traceback.format_exc()}")
                raise
        
        return frame_paths

    def create_video(self, frame_rate: int = 24, output_name: str = "animation.mp4") -> Optional[Path]:
        """Create video with proper encoding settings
        
        Args:
            frame_rate: Frames per second
            output_name: Output video filename
            
        Returns:
            Path to output video file or None if creation fails
        """
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
            if hasattr(self, 'temp_input'):
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

# Example usage:
if __name__ == "__main__":
    # Test animation generation
    test_image = "input.png"
    if Path(test_image).exists():
        processor = AnimationProcessor(test_image)
        try:
            # Test with preset
            processor.generate_frames(preset_name="glitch_surge", num_frames=30)
            video_path = processor.create_video(frame_rate=24, output_name="test_animation.mp4")
            if video_path:
                print(f"Animation created successfully: {video_path}")
        finally:
            processor.cleanup()
    else:
        print(f"Test image not found: {test_image}")