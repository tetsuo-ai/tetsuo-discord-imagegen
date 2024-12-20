import colorsys
import math
import os
import random
from bisect import bisect_right
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from dotenv import load_dotenv
from PIL import (
    Image,
    ImageChops,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
    ImageFont,
    ImageOps,
    ImageStat,
)
from scipy.ndimage import gaussian_filter

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
IMAGES_FOLDER = "images"
INPUT_IMAGE = "input.png"


def colorize_non_white(processor, color=(255, 0, 0)):  # Red as default color
    # Open the image
    img = processor
    # Convert to RGBA if not already in that mode
    img = img.convert("RGBA")
    datas = img.getdata()

    new_data = []
    for item in datas:
        # Check if the pixel isn't white (you might want to adjust this threshold)
        if item[:3] != (255, 255, 255):  # Assuming white is 255, 255, 255
            new_data.append(color + (item[3],))  # Keep the original alpha
        else:
            new_data.append(item)

    # Create new image with modified data
    img.putdata(new_data)
    return img


@dataclass
class Keyframe:
    """Represents a keyframe with time and value"""

    time: float  # 0.0 to 1.0
    value: Any
    easing: str = "linear"  # Options: 'linear', 'ease_in', 'ease_out', 'ease_in_out'

    def __post_init__(self):
        if not 0 <= self.time <= 1:
            raise ValueError("Keyframe time must be between 0 and 1")


class ParameterSchedule:
    """Represents a parameter that can have multiple keyframes"""

    def __init__(self, keyframes: Union[List[Keyframe], Any]):
        """Initialize with either keyframes or a static value"""
        if isinstance(keyframes, list) and all(
            isinstance(k, Keyframe) for k in keyframes
        ):
            self.keyframes = sorted(keyframes, key=lambda k: k.time)
        else:
            # If not given keyframes, treat as static value
            self.keyframes = [Keyframe(0.0, keyframes)]

    def _ease(self, progress: float, easing: str) -> float:
        """Apply easing function to progress"""
        if easing == "linear":
            return progress
        elif easing == "ease_in":
            return progress * progress
        elif easing == "ease_out":
            return 1 - (1 - progress) * (1 - progress)
        elif easing == "ease_in_out":
            return 0.5 * (math.sin((progress - 0.5) * math.pi) + 1)
        return progress

    def _interpolate_values(self, start_val: Any, end_val: Any, progress: float) -> Any:
        """Interpolate between two values"""
        if isinstance(start_val, (int, float)) and isinstance(end_val, (int, float)):
            return start_val + (end_val - start_val) * progress

        if isinstance(start_val, tuple) and isinstance(end_val, tuple):
            if len(start_val) != len(end_val):
                raise ValueError("Tuple values must have same length")
            return tuple(
                start + (end - start) * progress
                for start, end in zip(start_val, end_val)
            )

        return start_val  # Default to start value if types don't match

    def get_value(self, time: float) -> Any:
        """Get interpolated value at given time"""
        if time <= self.keyframes[0].time:
            return self.keyframes[0].value
        if time >= self.keyframes[-1].time:
            return self.keyframes[-1].value

        # Find surrounding keyframes
        idx = bisect_right([k.time for k in self.keyframes], time)
        k1, k2 = self.keyframes[idx - 1], self.keyframes[idx]

        # Calculate progress between keyframes
        segment_progress = (time - k1.time) / (k2.time - k1.time)
        eased_progress = self._ease(segment_progress, k1.easing)

        return self._interpolate_values(k1.value, k2.value, eased_progress)


class EffectParameters:
    """Handler for effect parameters supporting keyframe schedules"""

    def __init__(self, params: Dict[str, Any]):
        self.params = {}
        self._parse_params(params)

    def _parse_params(self, params: Dict[str, Any]):
        """Parse input parameters into ParameterSchedule objects"""
        for key, value in params.items():
            if isinstance(value, list) and all(isinstance(v, Keyframe) for v in value):
                # Already a list of keyframes
                self.params[key] = ParameterSchedule(value)
            elif isinstance(value, (list, tuple)) and len(value) == 2:
                # Simple start/end range - convert to keyframes
                if all(isinstance(v, (int, float)) for v in value):
                    keyframes = [Keyframe(0.0, value[0]), Keyframe(1.0, value[1])]
                    self.params[key] = ParameterSchedule(keyframes)
                else:
                    # Treat as regular value (e.g., color tuple)
                    self.params[key] = ParameterSchedule(value)
            else:
                # Static value
                self.params[key] = ParameterSchedule(value)

    def get_params_at_time(self, time: float) -> Dict[str, Any]:
        """Get all parameter values at given time"""
        return {key: schedule.get_value(time) for key, schedule in self.params.items()}


# Effect order and presets
EFFECT_ORDER = [
    "impact",
    "rgb",
    "color",
    "glitch",
    "chroma",
    "scan",
    "noise",
    "energy",
    "pulse",
    "consciousness",
    "channel_pass",
]

# Example of a complex animation preset using keyframes

ANIMATION_PRESETS = {
    "cyberpunk": {
        "params": {
            "glitch": (3, 8),  # Start and end values
            "chroma": (8, 15),
            "scan": (50, 150),
            "consciousness": (0.3, 0.7),
        },
        "frames": 30,
        "fps": 24,
        "description": "Cyberpunk-style glitch effect with scanning",
    },
    "vaporwave": {
        "params": {
            "chroma": (10, 20),
            "scan": (100, 200),
            "rgb": ((255, 100, 255), (100, 255, 255)),
            "rgbalpha": (150, 200),
        },
        "frames": 30,
        "fps": 24,
        "description": "Vaporwave aesthetic with color shifts",
    },
    "glitch_art": {
        "params": {"glitch": (5, 15), "chroma": (12, 25), "noise": (0.1, 0.3)},
        "frames": 30,
        "fps": 24,
        "description": "Heavy glitch effects with chromatic aberration",
    },
    "matrix": {
        "params": {
            "scan": (50, 150),
            "consciousness": (0.4, 0.8),
            "rgb": ((0, 255, 0), (50, 255, 50)),
            "rgbalpha": (150, 200),
        },
        "frames": 30,
        "fps": 24,
        "description": "Matrix-inspired digital rain effect",
    },
    "psychic": {
        "params": {
            "consciousness": (0.3, 0.9),
            "energy": (0.2, 0.7),
            "pulse": (0.1, 0.6),
        },
        "frames": 30,
        "fps": 24,
        "description": "Psychic energy visualization",
    },
    "akira": {
        "params": {
            "consciousness": (0.5, 0.9),
            "energy": (0.3, 0.8),
            "pulse": (0.2, 0.7),
            "glitch": (2, 6),
        },
        "frames": 30,
        "fps": 24,
        "description": "Akira-inspired psychic power effect",
    },
}


class ImageAnalyzer:

    @staticmethod
    def analyze_image(image: Image.Image) -> Dict[str, float]:
        """Comprehensive image analysis for adaptive processing"""
        if image.mode != "RGB":
            image = image.convert("RGB")

        stat = ImageStat.Stat(image)
        brightness = sum(stat.mean) / (3 * 255.0)
        contrast = sum(stat.stddev) / (3 * 255.0)

        r, g, b = stat.mean
        color_variance = np.std([r, g, b])

        edges = image.filter(ImageFilter.FIND_EDGES)
        edge_stat = ImageStat.Stat(edges)
        complexity = sum(edge_stat.mean) / (3 * 255.0)

        return {
            "brightness": brightness,
            "contrast": contrast,
            "color_variance": color_variance,
            "complexity": complexity,
        }

    @staticmethod
    def get_adaptive_params(
        analysis: Dict[str, float],
        user_params: Optional[Dict[str, Any]] = None,
        preset_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate adaptive parameters based on image analysis"""
        params = {}
        requested_effects = set()

        if user_params:
            requested_effects.update(user_params.keys())
        if preset_params:
            requested_effects.update(preset_params.keys())

        if "coloralpha" in requested_effects:
            params["coloralpha"] = int(
                255
                * (1.0 - (analysis["brightness"] * 0.7 + analysis["contrast"] * 0.3))
            )

        if "glitch" in requested_effects:
            params["glitch"] = int(20 * (1.0 - (analysis["complexity"] * 0.8)))

        if "chroma" in requested_effects:
            params["chroma"] = int(
                20 * (1.0 - (analysis["color_variance"] / 255 * 0.9))
            )

        if "noise" in requested_effects:
            params["noise"] = float(
                analysis["contrast"] * 0.5 + analysis["brightness"] * 0.5
            )

        if "scan" in requested_effects:
            params["scan"] = int(10 * (1.0 - analysis["complexity"]))

        return params


def offset_channel(image: Image.Image, offset_x: int, offset_y: int) -> Image.Image:
    """Enhanced channel offset with smoother transitions"""
    width, height = image.size
    offset_image = Image.new(image.mode, (width, height), 0)

    if offset_x > 0:
        left_part = image.crop((width - offset_x, 0, width, height))
        main_part = image.crop((0, 0, width - offset_x, height))
        offset_image.paste(left_part, (0, 0))
        offset_image.paste(main_part, (offset_x, 0))
    elif offset_x < 0:
        right_part = image.crop((0, 0, -offset_x, height))
        main_part = image.crop((-offset_x, 0, width, height))
        offset_image.paste(right_part, (width + offset_x, 0))
        offset_image.paste(main_part, (0, 0))
    else:
        offset_image = image.copy()

    # Apply Gaussian Blur
    offset_image = offset_image.filter(ImageFilter.GaussianBlur(0.5))

    # Ensure the resulting image has the same size as the original
    offset_image = offset_image.resize((width, height), Image.Resampling.LANCZOS)

    return offset_image


class ImageProcessor:
    def __init__(
        self, image_input: Union[str, bytes, Image.Image, BytesIO], points: bool = False
    ):
        if isinstance(image_input, str):
            self.base_image = Image.open(image_input)
        elif isinstance(image_input, bytes):
            self.base_image = Image.open(BytesIO(image_input))
        elif isinstance(image_input, Image.Image):
            self.base_image = image_input
        elif isinstance(image_input, BytesIO):
            self.base_image = Image.open(image_input)
        else:
            raise ValueError("Unsupported image input type")

    def apply_glitch(self, intensity: float) -> Image.Image:
        """Apply glitch effect with given intensity"""
        img = self.base_image.copy()
        img = img.convert("RGB")
        arr = np.array(img)

        # Number of glitch lines based on intensity
        num_lines = int(intensity * 5)
        height = arr.shape[0]

        for _ in range(num_lines):
            # Random line position and offset
            y = random.randint(0, height - 1)
            offset = random.randint(-10, 10)

            # Shift line horizontally
            if 0 <= y < height:
                arr[y, :] = np.roll(arr[y, :], offset, axis=0)

        return Image.fromarray(arr)

    # Make the memes!

    def apply_impact_text(self, text: str) -> Image.Image:
        # Open the image and convert it to RGBA mode
        impact_image = self.base_image.copy().convert("RGBA")
        # Create a new image for the text overlay with transparency
        txt = Image.new("RGBA", impact_image.size, (255, 255, 255, 0))
        # Load the font
        font = ImageFont.truetype("impact.ttf", 70)
        # Draw context
        d = ImageDraw.Draw(txt)
        # Define the text and colors
        outline_color = (0, 0, 0, 255)  # Black with full opacity
        text_color = (255, 255, 255, 255)  # White with full opacity
        # Calculate text size to center it
        text_bbox = d.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        position = (
            (impact_image.width - text_width) // 2,
            (impact_image.height - text_height) // 1.15,
        )
        # Draw outline by drawing the text shifted slightly in all directions
        for outline_offset in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            d.text(
                (position[0] + outline_offset[0], position[1] + outline_offset[1]),
                text,
                fill=outline_color,
                font=font,
            )
            # Draw the main text over the outline
            d.text(position, text, fill=text_color, font=font)
        # Combine the text overlay with the original impact_image
        combined = Image.alpha_composite(impact_image, txt)
        return combined

    def apply_chromatic_aberration(self, offset: float) -> Image.Image:
        """Apply RGB channel offset"""
        img = self.base_image.copy()
        img = img.convert("RGB")
        r, g, b = img.split()

        # Create offset versions of channels
        r = ImageOps.expand(r, border=(int(offset), 0, 0, 0), fill=0)
        b = ImageOps.expand(b, border=(0, 0, int(offset), 0), fill=0)

        # Crop to original size
        width, height = img.size
        r = r.crop((0, 0, width, height))
        b = b.crop((0, 0, width, height))

        # Merge channels
        return Image.merge("RGB", (r, g, b))

    def apply_scan_lines(self, gap: float) -> Image.Image:
        """Apply scan line effect"""
        img = self.base_image.copy()
        img = img.convert("RGB")
        width, height = img.size
        draw = ImageDraw.Draw(img)

        # Draw horizontal lines
        for y in range(0, height, max(1, int(gap))):
            draw.line([(0, y), (width, y)], fill=(0, 0, 0), width=1)

        return img

    def apply_noise(self, intensity: float) -> Image.Image:
        """Add noise to image"""
        img = self.base_image.copy()
        img = img.convert("RGB")
        arr = np.array(img)

        # Generate noise
        noise = np.random.normal(0, intensity * 255, arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(noisy)

    def apply_energy_effect(self, intensity: float) -> Image.Image:
        """Apply energy distortion effect"""
        img = self.base_image.copy()
        img = img.convert("RGB")
        arr = np.array(img)

        # Create energy distortion
        x = np.arange(arr.shape[1])
        y = np.arange(arr.shape[0])
        X, Y = np.meshgrid(x, y)

        distortion = np.sin(X * 0.1 + Y * 0.1) * intensity * 30

        # Apply distortion to each channel
        for c in range(3):
            arr[:, :, c] = np.clip(arr[:, :, c] + distortion, 0, 255)

        return Image.fromarray(arr.astype(np.uint8))

    def apply_pulse_effect(self, intensity: float) -> Image.Image:
        """Apply pulsing effect"""
        img = self.base_image.copy()
        img = img.convert("RGB")

        # Enhance brightness based on intensity
        enhancer = ImageEnhance.Brightness(img)
        pulse_factor = 1.0 + intensity
        return enhancer.enhance(pulse_factor)

    def apply_consciousness(self, intensity: float) -> Image.Image:
        """Apply consciousness effect (combination of effects)"""
        img = self.base_image.copy()

        # Apply multiple effects in sequence
        img = self.apply_energy_effect(intensity * 0.5)
        img = self.apply_pulse_effect(intensity * 0.3)
        if intensity > 0.5:
            img = self.apply_chromatic_aberration(intensity * 5)

        return img

    def add_color_overlay(self, color: Tuple[int, int, int, int]) -> Image.Image:
        """Add color overlay with alpha"""
        img = self.base_image.copy()
        img = img.convert("RGBA")

        # Create color overlay
        overlay = Image.new("RGBA", img.size, color)

        # Blend images
        return Image.alpha_composite(img, overlay)

    def convertImageToAscii(
        self, cols: int = 80, scale: float = 0.43, moreLevels: bool = False
    ) -> List[str]:
        """Convert image to ASCII art"""
        # Define ASCII characters
        chars = (
            np.asarray(list(" .,:;irsXA253hMHGS#9B&@"))
            if moreLevels
            else np.asarray(list(" .:-=+*#%@"))
        )

        # Calculate dimensions
        img = self.base_image.copy()
        W, H = img.size
        w = W / cols
        h = w / scale
        rows = int(H / h)

        # Resize image
        if cols > W or rows > H:
            raise ValueError("Image too small for specified columns!")

        img = img.resize((cols, rows), Image.Resampling.LANCZOS)
        img = img.convert("L")  # Convert to grayscale

        # Map pixels to characters
        pixels = np.array(img)
        result = []
        for row in range(rows):
            line = ""
            for col in range(cols):
                pixel_value = pixels[row, col]
                # Map pixel value to character index
                char_idx = (pixel_value * (len(chars) - 1) / 255).astype(int)
                line += chars[char_idx]
            result.append(line)

        return result

    def apply_effect(self, effect_name: str, params: dict) -> Image.Image:
        """Apply an effect with parameters"""
        if effect_name not in params:
            return self.base_image

        value = params[effect_name]

        # Handle tuple values for animations
        if isinstance(value, tuple):
            # For static images, use the first value
            if isinstance(value[0], (int, float)):
                value = value[0]
            elif isinstance(value[0], tuple):  # For RGB values
                value = value[0]

        if effect_name == "impact":
            return self.apply_impact_text(value)
        if effect_name == "glitch":
            return self.apply_glitch(value)
        elif effect_name == "chroma":
            return self.apply_chromatic_aberration(value)
        elif effect_name == "scan":
            return self.apply_scan_lines(value)
        elif effect_name == "noise":
            return self.apply_noise(value)
        elif effect_name == "energy":
            return self.apply_energy_effect(value)
        elif effect_name == "pulse":
            return self.apply_pulse_effect(value)
        elif effect_name == "consciousness":
            return self.apply_consciousness(value)
        elif effect_name == "rgb":
            if isinstance(value, tuple):
                return self.add_color_overlay((*value, params.get("rgbalpha", 255)))

        return self.base_image
