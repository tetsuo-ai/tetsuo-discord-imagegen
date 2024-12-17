import numpy as np
from PIL import Image, ImageEnhance, ImageChops
from pathlib import Path
import subprocess
from tetimi import ImageProcessor, EFFECT_PRESETS
import tempfile
import shutil
import argparse
import random
import colorsys
import logging
import traceback
import os
import gc

# Animation presets 
ANIMATION_PRESETS = {
    'glitch_surge': {
        'start_glitch': 1,
        'end_glitch': 25,
        'chroma': (3, 15),
        'scan': (60, 120),
        'noise': (0.02, 0.08)
    },
    'power_surge': {
        'energy': (0.2, 0.8),
        'pulse': (0.1, 0.4),
        'chroma': (5, 12),
        'noise': (0.01, 0.05)
    },
    'psychic_blast': {
        'pulse': (0.3, 0.9),
        'energy': (0.4, 0.9),
        'color_shift': True,
        'noise': (0.02, 0.06)
    },
    'digital_decay': {
        'glitch': (5, 15),
        'chroma': (8, 20),
        'scan': (80, 160),
        'noise': (0.03, 0.09)
    },
    'neo_flash': {
        'pulse': (0.2, 0.7),
        'color_shift': True,
        'scan': (100, 200),
        'energy': (0.3, 0.6)
    }
}


class AnimationProcessor:
    def __init__(self, input_image: str, output_dir: str = "animations"):
        """Initialize with enhanced logging and validation"""
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger('AnimationProcessor')
        
        self.input_image = Path(input_image)
        if not self.input_image.exists():
            raise ValueError(f"Input image not found: {input_image}")
            
        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / "frames"
        
        # Create directories with logging
        self.logger.debug(f"Setting up directories: {self.output_dir}, {self.frames_dir}")
        self.output_dir.mkdir(exist_ok=True)
        self.frames_dir.mkdir(exist_ok=True)
        
        # Validate input image can be opened
        try:
            with Image.open(self.input_image) as img:
                self.logger.debug(f"Successfully opened input image: {input_image}")
        except Exception as e:
            self.logger.error(f"Failed to open input image: {e}")
            raise
            
    def generate_color_shift(self, num_frames: int) -> list:
        """Generate smooth color transition sequence"""
        hue_start = random.random()
        hue_shift = random.uniform(0.2, 0.8)
        hue_sequence = np.linspace(hue_start, hue_start + hue_shift, num_frames) % 1.0
        
        return [
            tuple(int(x * 255) for x in colorsys.hsv_to_rgb(h, 0.9, 0.9))
            for h in hue_sequence
        ]
        

    def apply_preset_frame(self, processor: ImageProcessor, preset: dict, progress: float) -> Image.Image:
        def interpolate_value(start, end, progress):
            if isinstance(start, tuple) and isinstance(end, tuple):
                return tuple(
                    start[i] + (end[i] - start[i]) * progress 
                    for i in range(len(start))
                )
            elif isinstance(start, (int, float)) and isinstance(end, (int, float)):
                return start + (end - start) * progress
            else:
                raise TypeError("Incompatible types for interpolation")
        
        result = processor.base_image.copy().convert('RGBA')
        
        effects = [
            ('start_glitch', 'end_glitch', processor.apply_glitch_effect, int, (1, 40)),
            ('chroma', 'chroma', processor.add_chromatic_aberration, int, (1, 40)),
            ('scan', 'scan', processor.add_scan_lines, int, (1, 120)),
            ('noise', 'noise', processor.add_noise, float, (0, 1))
        ]
        
        for start_key, end_key, effect_func, value_type, valid_range in effects:
            if start_key in preset and end_key in preset:
                start, end = preset[start_key], preset[end_key]
                
                interpolated_value = interpolate_value(start, end, progress)
                
                if isinstance(interpolated_value, tuple):
                    interpolated_value = tuple(
                        max(valid_range[0], min(valid_range[1], v)) 
                        for v in interpolated_value
                    )
                else:
                    interpolated_value = max(valid_range[0], min(valid_range[1], interpolated_value))
                
                result = effect_func(int(interpolated_value) if isinstance(interpolated_value, (int, float)) else interpolated_value)
        
        return result


    def generate_frames(self, preset_name: str = None, num_frames: int = 60) -> list:
        preset = ANIMATION_PRESETS.get(preset_name, ANIMATION_PRESETS['glitch_surge'])
        frame_paths = []
        
        for i in range(num_frames):
            progress = i / (num_frames - 1)
            
            try:
                processor = ImageProcessor(str(self.input_image))
                result = self.apply_preset_frame(processor, preset, progress)
                
                frame_path = self.frames_dir / f"frame_{i:04d}.png"
                result.save(frame_path)
                
                # Explicitly delete references
                del result
                gc.collect()  # Force garbage collection
                
                frame_paths.append(frame_path)
                
            except Exception as e:
                self.logger.error(f"Frame generation error: {traceback.format_exc()}")
                raise
        
        return frame_paths
    def create_video(self, frame_rate: int = 24, output_name: str = "animation.mp4") -> Path:
        """Create video with enhanced error checking"""
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
            str(output_path)
        ]
        
        self.logger.debug(f"Running ffmpeg command: {' '.join(ffmpeg_cmd)}")
        
        try:
            result = subprocess.run(ffmpeg_cmd, 
                                  check=True,
                                  capture_output=True,
                                  text=True)
            self.logger.debug(f"ffmpeg output: {result.stdout}")
            return output_path
        except subprocess.CalledProcessError as e:
            self.logger.error(f"ffmpeg error: {e.stderr}")
            return None
    def cleanup(self):
        """Remove temporary frame files after video creation"""
        if self.frames_dir.exists():
            shutil.rmtree(self.frames_dir)
def main():
    parser = argparse.ArgumentParser(description='Generate effect animations')
    parser.add_argument('--input', default='input.png', help='Input image path')
    parser.add_argument('--style', default='glitch_surge', 
                        choices=ANIMATION_PRESETS.keys(),
                        help='Animation preset to use')
    parser.add_argument('--frames', type=int, default=60, 
                        help='Number of frames')
    parser.add_argument('--fps', type=int, default=24, 
                        help='Frames per second')
    parser.add_argument('--output-dir', default='animations', 
                        help='Output directory')
    parser.add_argument('--format', choices=['mp4', 'gif', 'both'],
                        default='mp4', help='Output format')
    
    args = parser.parse_args()
    
    processor = AnimationProcessor(args.input, args.output_dir)
    processor.generate_frames(args.preset, args.frames)
    
    if args.format in ['mp4', 'both']:
        processor.create_video(args.fps)
    if args.format in ['gif', 'both']:
        processor.create_gif(args.fps)
        
    processor.cleanup()

if __name__ == "__main__":
    main()