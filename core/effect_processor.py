# image_processing/core/effect_processor.py

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union, List
from .image_processor import BaseImageProcessor
import math

class EffectProcessor(BaseImageProcessor):
    """
    Extends BaseImageProcessor with specific effect implementations.
    """
    
    def apply_glitch(self, intensity: float) -> None:
        """
        Apply glitch effect with given intensity.
        
        Args:
            intensity: Effect intensity (0.0 to 1.0)
        """
        if not 0 <= intensity <= 1:
            raise ValueError("Intensity must be between 0 and 1")
            
        self.ensure_rgb()
        self.history.append(self.current_image.copy())
        
        # Convert to numpy array for efficient processing
        arr = np.array(self.current_image)
        height = arr.shape[0]
        
        # Number of glitch lines based on intensity
        num_lines = int(intensity * height * 0.1)  # Up to 10% of height
        
        for _ in range(num_lines):
            # Random line position and offset
            y = np.random.randint(0, height-1)
            offset = np.random.randint(-int(intensity * 20), int(intensity * 20))
            
            # Shift line horizontally
            if 0 <= y < height:
                arr[y, :] = np.roll(arr[y, :], offset, axis=0)
        
        self.current_image = Image.fromarray(arr)

    def apply_chromatic_aberration(self, offset: float) -> None:
        """
        Apply RGB channel offset (chromatic aberration).
        
        Args:
            offset: Maximum pixel offset (0.0 to 1.0)
        """
        if not 0 <= offset <= 1:
            raise ValueError("Offset must be between 0 and 1")
            
        self.ensure_rgb()
        self.history.append(self.current_image.copy())
        
        # Calculate pixel offsets based on image size
        width, _ = self.current_image.size
        max_offset = int(width * offset * 0.1)  # Up to 10% of width
        
        # Offset red channel left, blue channel right
        self.offset_channel('R', -max_offset)
        self.offset_channel('B', max_offset)

    def apply_scan_lines(self, gap: int, opacity: float = 0.5) -> None:
        """
        Apply scan line effect.
        
        Args:
            gap: Pixels between scan lines
            opacity: Line opacity (0.0 to 1.0)
        """
        if gap < 1:
            raise ValueError("Gap must be at least 1 pixel")
        if not 0 <= opacity <= 1:
            raise ValueError("Opacity must be between 0 and 1")
            
        self.history.append(self.current_image.copy())
        width, height = self.current_image.size
        
        # Create scan line overlay
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Draw horizontal lines
        for y in range(0, height, gap):
            alpha = int(opacity * 255)
            draw.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))
        
        # Composite with original
        self.ensure_rgba()
        self.current_image = Image.alpha_composite(self.current_image, overlay)

    def apply_noise(self, intensity: float) -> None:
        """
        Add noise to image.
        
        Args:
            intensity: Noise intensity (0.0 to 1.0)
        """
        if not 0 <= intensity <= 1:
            raise ValueError("Intensity must be between 0 and 1")
            
        self.ensure_rgb()
        self.history.append(self.current_image.copy())
        
        # Convert to numpy array
        arr = np.array(self.current_image)
        
        # Generate noise
        noise = np.random.normal(0, intensity * 50, arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        
        self.current_image = Image.fromarray(noisy)

    def apply_energy_effect(self, intensity: float) -> None:
        """
        Apply energy distortion effect.
        
        Args:
            intensity: Effect intensity (0.0 to 1.0)
        """
        if not 0 <= intensity <= 1:
            raise ValueError("Intensity must be between 0 and 1")
            
        self.ensure_rgb()
        self.history.append(self.current_image.copy())
        
        # Convert to numpy array
        arr = np.array(self.current_image)
        
        # Create displacement map
        x = np.arange(arr.shape[1])
        y = np.arange(arr.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # Generate distortion pattern
        distortion = np.sin(X * 0.1 + Y * 0.1) * intensity * 30
        
        # Apply to each channel
        for c in range(3):
            arr[:,:,c] = np.clip(arr[:,:,c] + distortion, 0, 255)
        
        self.current_image = Image.fromarray(arr.astype(np.uint8))

    def apply_pulse_effect(self, intensity: float) -> None:
        """
        Apply pulsing effect through brightness modulation.
        
        Args:
            intensity: Effect intensity (0.0 to 1.0)
        """
        if not 0 <= intensity <= 1:
            raise ValueError("Intensity must be between 0 and 1")
        
        self.history.append(self.current_image.copy())
        enhancer = ImageEnhance.Brightness(self.current_image)
        pulse_factor = 1.0 + intensity
        self.current_image = enhancer.enhance(pulse_factor)

    def apply_consciousness_effect(self, intensity: float) -> None:
        """
        Apply consciousness effect (combination of energy, pulse, and chromatic aberration).
        
        Args:
            intensity: Effect intensity (0.0 to 1.0)
        """
        if not 0 <= intensity <= 1:
            raise ValueError("Intensity must be between 0 and 1")
        
        self.history.append(self.current_image.copy())
        
        # Apply multiple effects in sequence
        self.apply_energy_effect(intensity * 0.5)
        self.apply_pulse_effect(intensity * 0.3)
        
        if intensity > 0.5:
            self.apply_chromatic_aberration(intensity * 0.4)

    def apply_effect(self, effect_name: str, params: Dict[str, Any]) -> None:
        """
        Apply a named effect with parameters.
        
        Args:
            effect_name: Name of the effect to apply
            params: Dictionary of effect parameters
        """
        effect_map = {
            'glitch': lambda p: self.apply_glitch(p.get('intensity', 0.5)),
            'chroma': lambda p: self.apply_chromatic_aberration(p.get('offset', 0.5)),
            'scan': lambda p: self.apply_scan_lines(
                p.get('gap', 2), 
                p.get('opacity', 0.5)
            ),
            'noise': lambda p: self.apply_noise(p.get('intensity', 0.5)),
            'energy': lambda p: self.apply_energy_effect(p.get('intensity', 0.5)),
            'pulse': lambda p: self.apply_pulse_effect(p.get('intensity', 0.5)),
            'consciousness': lambda p: self.apply_consciousness_effect(p.get('intensity', 0.5))
        }
        
        if effect_name not in effect_map:
            raise ValueError(f"Unknown effect: {effect_name}")
            
        effect_map[effect_name](params)

    def apply_effects_sequence(self, effects: List[Tuple[str, Dict[str, Any]]]) -> None:
        """
        Apply a sequence of effects in order.
        
        Args:
            effects: List of (effect_name, parameters) tuples
        """
        for effect_name, params in effects:
            self.apply_effect(effect_name, params)

    def create_effect_animation(self, effect_name: str, 
                              params: Dict[str, Any],
                              num_frames: int = 30) -> List[Image.Image]:
        """
        Create animation frames for an effect.
        
        Args:
            effect_name: Name of the effect to animate
            params: Effect parameters including start/end values
            num_frames: Number of frames to generate
        
        Returns:
            List[Image.Image]: List of animation frames
        """
        frames = []
        
        for i in range(num_frames):
            # Calculate progress through animation
            progress = i / (num_frames - 1)
            
            # Create frame parameters by interpolating between start/end values
            frame_params = {}
            for param, value in params.items():
                if isinstance(value, tuple) and len(value) == 2:
                    start, end = value
                    frame_params[param] = start + (end - start) * progress
                else:
                    frame_params[param] = value
            
            # Create frame
            frame_processor = EffectProcessor(self.original_image)
            frame_processor.apply_effect(effect_name, frame_params)
            frames.append(frame_processor.get_current_image())
        
        return frames