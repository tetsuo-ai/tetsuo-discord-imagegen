import numpy as np
from PIL import Image, ImageEnhance, ImageChops, ImageFilter
from pathlib import Path
import subprocess
import tempfile
import shutil
import random
import colorsys
import logging
import traceback
import os
import gc
from typing import Optional, Tuple, Dict, Any, List, Union
from io import BytesIO
from tetimi import EFFECT_ORDER

# Animation presets with enhanced parameters
ANIMATION_PRESETS = {
    'glitch_surge': {
        'glitch': (1, 25),
        'chroma': (3, 15),
        'scan': (60, 120),
        'noise': (0.02, 0.08),
        'energy': (0.1, 0.4)
    },
    'power_surge': {
        'energy': (0.2, 0.8),
        'pulse': (0.1, 0.4),
        'chroma': (5, 12),
        'noise': (0.01, 0.05),
        'scan': (40, 90)
    },
    'psychic_blast': {
        'pulse': (0.3, 0.9),
        'energy': (0.4, 0.9),
        'color_shift': True,
        'noise': (0.02, 0.06),
        'chroma': (8, 18)
    },
    'digital_decay': {
        'glitch': (5, 15),
        'chroma': (8, 20),
        'scan': (80, 160),
        'noise': (0.03, 0.09),
        'energy': (0.2, 0.5)
    },
    'neo_flash': {
        'pulse': (0.2, 0.7),
        'color_shift': True,
        'scan': (100, 200),
        'energy': (0.3, 0.6),
        'chroma': (10, 25)
    }
}

class AnimationProcessor:
    def __init__(self, image_input: Union[str, bytes, Image.Image, BytesIO], output_dir: str = "animations"):
        """
        Initialize animation processor with enhanced input handling
        
        Args:
            image_input: Input image in various formats
            output_dir: Directory for output files
        """
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger('AnimationProcessor')
        
        # Handle different input types
        if isinstance(image_input, str):
            self.input_path = Path(image_input)
            if not self.input_path.exists():
                raise ValueError(f"Input image not found: {image_input}")
        else:
            # For non-path inputs, save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                if isinstance(image_input, bytes):
                    tmp.write(image_input)
                elif isinstance(image_input, Image.Image):
                    image_input.save(tmp.name, format='PNG')
                elif isinstance(image_input, BytesIO):
                    tmp.write(image_input.getvalue())
                self.input_path = Path(tmp.name)
                self.temp_input = True
        
        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / "frames"
        
        # Create directories
        self.logger.debug(f"Setting up directories: {self.output_dir}, {self.frames_dir}")
        self.output_dir.mkdir(exist_ok=True)
        self.frames_dir.mkdir(exist_ok=True)
        
        # Initialize base ImageProcessor
        from tetimi import ImageProcessor
        self.base_processor = ImageProcessor(str(self.input_path))
        
    def generate_color_shift(self, num_frames: int) -> List[tuple]:
        """Generate smooth color transition sequence"""
        hue_start = random.random()
        hue_shift = random.uniform(0.2, 0.8)
        hue_sequence = np.linspace(hue_start, hue_start + hue_shift, num_frames) % 1.0
        
        return [
            tuple(int(x * 255) for x in colorsys.hsv_to_rgb(h, 0.9, 0.9))
            for h in hue_sequence
        ]

    def interpolate_value(self, start: Union[int, float, tuple], 
                         end: Union[int, float, tuple], 
                         progress: float) -> Union[int, float, tuple]:
        """Interpolate between values or tuples"""
        if isinstance(start, tuple) and isinstance(end, tuple):
            return tuple(
                start[i] + (end[i] - start[i]) * progress 
                for i in range(len(start))
            )
        return start + (end - start) * progress

    def apply_frame_effects(self, processor: Any, preset: Dict[str, Any], progress: float) -> Image.Image:
        result = processor.base_image.copy()
        
        for effect in EFFECT_ORDER:
            if effect not in preset:
                continue
                
            value = preset[effect]
            if isinstance(value, tuple):
                # Value is (start, end) for animation
                start, end = value 
                current_value = start + (end - start) * progress
            else:
                current_value = value
                
            processor.base_image = result
            result = processor.apply_effect(effect, {effect: current_value})
        
        return result

    def generate_frames(self, preset_name: str = None, num_frames: int = 60) -> List[Path]:
        """Generate animation frames using presets"""
        preset = ANIMATION_PRESETS.get(preset_name, ANIMATION_PRESETS['glitch_surge'])
        frame_paths = []
        
        # Add color shift if specified
        if preset.get('color_shift'):
            colors = self.generate_color_shift(num_frames)
            preset['color'] = list(zip(colors[:-1], colors[1:]))
        
        for i in range(num_frames):
            progress = i / (num_frames - 1)
            
            try:
                # Create frame-specific processor
                processor = self.base_processor
                result = self.apply_frame_effects(processor, preset, progress)
                
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
        """Create video from frames"""
        self.logger.info(f"Creating video with frame rate: {frame_rate}")
        output_path = self.output_dir / output_name
        
        # Verify frames exist
        frame_files = list(self.frames_dir.glob("frame_*.png"))
        if not frame_files:
            self.logger.error("No frames found for video creation")
            raise ValueError("No frames found for video creation")
            
        self.logger.debug(f"Found {len(frame_files)} frames")
        
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-framerate', str(frame_rate),
            '-i', str(self.frames_dir / 'frame_%04d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            '-preset', 'medium',  # Balance between speed and quality
            '-movflags', '+faststart',  # Enable fast start for web playback
            str(output_path)
        ]
        
        self.logger.debug(f"Running ffmpeg command: {' '.join(ffmpeg_cmd)}")
        
        try:
            result = subprocess.run(
                ffmpeg_cmd,
                check=True,
                capture_output=True,
                text=True
            )
            self.logger.debug(f"ffmpeg output: {result.stdout}")
            return output_path
        except subprocess.CalledProcessError as e:
            self.logger.error(f"ffmpeg error: {e.stderr}")
            return None
        finally:
            # Cleanup temporary input file if it exists
            if hasattr(self, 'temp_input'):
                os.unlink(self.input_path)

    def cleanup(self):
        """Remove temporary files"""
        if self.frames_dir.exists():
            shutil.rmtree(self.frames_dir)
            
    def __del__(self):
        """Cleanup on object destruction"""
        self.cleanup()


# Example usage
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate effect animations')
    parser.add_argument('--input', required=True, help='Input image path')
    parser.add_argument('--style', default='glitch_surge', 
                       choices=ANIMATION_PRESETS.keys(),
                       help='Animation preset to use')
    parser.add_argument('--frames', type=int, default=60, 
                       help='Number of frames')
    parser.add_argument('--fps', type=int, default=24, 
                       help='Frames per second')
    parser.add_argument('--output-dir', default='animations', 
                       help='Output directory')
    
    args = parser.parse_args()
    
    try:
        processor = AnimationProcessor(args.input, args.output_dir)
        processor.generate_frames(args.style, args.frames)
        output_path = processor.create_video(args.fps)
        
        if output_path:
            print(f"Animation created successfully: {output_path}")
        else:
            print("Failed to create animation")
            
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        processor.cleanup()

if __name__ == "__main__":
    main()