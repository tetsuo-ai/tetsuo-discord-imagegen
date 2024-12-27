import colorsys
import math
import random
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import (
    Image,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
    ImageFont,
    ImageOps,
    ImageStat,
)


class BaseImageProcessor:
    """
    Core image processing functionality that handles basic image operations
    and provides a foundation for more complex effects.
    """

    def __init__(
        self,
        image_input: Union[str, bytes, Image.Image, BytesIO, None],
        random_flag=False,
    ):
        """
        Initialize the image processor with flexible input handling.

        Args:
            image_input: Can be:
                - str: Path to image file
                - bytes: Raw image data
                - Image.Image: PIL Image object
                - BytesIO: BytesIO containing image data
        """
        self.original_image = self._load_image(image_input)
        self.current_image = self.original_image.copy()
        self.history: List[Image.Image] = []

    def _load_image(
        self, image_input: Union[str, bytes, Image.Image, BytesIO, None]
    ) -> Image.Image:
        """
        Load image from various input types.
        """
        if isinstance(image_input, str):
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
            raise ValueError(
                f"Unsupported image input type: \
                             {type(image_input)}"
            )

    def get_image_stats(self) -> Dict[str, float]:
        """
        Calculate basic image statistics for adaptive processing.
        """
        if self.current_image.mode != "RGB":
            analysis_image = self.current_image.convert("RGB")
        else:
            analysis_image = self.current_image

        stat = ImageStat.Stat(analysis_image)

        brightness = sum(stat.mean) / (3 * 255.0)
        contrast = sum(stat.stddev) / (3 * 255.0)

        r, g, b = stat.mean
        color_variance = np.std([r, g, b])

        edges = analysis_image.filter(ImageFilter.FIND_EDGES)
        edge_stat = ImageStat.Stat(edges)
        complexity = sum(edge_stat.mean) / (3 * 255.0)

        return {
            "brightness": brightness,
            "contrast": contrast,
            "color_variance": float(color_variance),
            "complexity": complexity,
        }

    def resize(
        self,
        size: Tuple[int, int],
        resample: Image.Resampling = Image.Resampling.LANCZOS,
    ) -> None:
        """
        Resize the current image.
        """
        self.history.append(self.current_image.copy())
        self.current_image = self.current_image.resize(size, resample)

    def ensure_rgb(self) -> None:
        """
        Ensure image is in RGB mode.
        """
        if self.current_image.mode != "RGB":
            self.history.append(self.current_image.copy())
            self.current_image = self.current_image.convert("RGB")

    def ensure_rgba(self) -> None:
        """
        Ensure image is in RGBA mode.
        """
        if self.current_image.mode != "RGBA":
            self.history.append(self.current_image.copy())
            self.current_image = self.current_image.convert("RGBA")

    def adjust_brightness(self, factor: float) -> None:
        """
        Adjust image brightness.

        Args:
            factor: Brightness adjustment factor (0.0 to 2.0)
        """
        self.history.append(self.current_image.copy())
        enhancer = ImageEnhance.Brightness(self.current_image)
        self.current_image = enhancer.enhance(factor)

    def adjust_contrast(self, factor: float) -> None:
        """
        Adjust image contrast.

        Args:
            factor: Contrast adjustment factor (0.0 to 2.0)
        """
        self.history.append(self.current_image.copy())
        enhancer = ImageEnhance.Contrast(self.current_image)
        self.current_image = enhancer.enhance(factor)

    def apply_blur(self, radius: float) -> None:
        """
        Apply Gaussian blur to the image.

        Args:
            radius: Blur radius
        """
        self.history.append(self.current_image.copy())
        self.current_image = self.current_image.filter(
            ImageFilter.GaussianBlur(radius=radius)
        )

    def apply_color_overlay(self, color: Tuple[int, int, int], alpha: int) -> None:
        """
        Apply a color overlay with transparency.

        Args:
            color: RGB color tuple
            alpha: Opacity (0-255)
        """
        self.ensure_rgba()
        self.history.append(self.current_image.copy())

        overlay = Image.new("RGBA", self.current_image.size, (*color, alpha))
        self.current_image = Image.alpha_composite(self.current_image, overlay)

    def get_channel(self, channel: str) -> Image.Image:
        """
        Get a specific color channel.

        Args:
            channel: 'R', 'G', 'B', or 'A'
        """
        if channel.upper() not in ["R", "G", "B", "A"]:
            raise ValueError("Channel must be 'R', 'G', 'B', or 'A'")

        if channel.upper() == "A" and "A" not in self.current_image.getbands():
            self.ensure_rgba()

        return self.current_image.getchannel(channel.upper())

    def set_channel(self, channel: str, data: Image.Image) -> None:
        """
        Set a specific color channel.

        Args:
            channel: 'R', 'G', 'B', or 'A'
            data: Single-channel image data
        """
        if channel.upper() not in ["R", "G", "B", "A"]:
            raise ValueError("Channel must be 'R', 'G', 'B', or 'A'")

        self.history.append(self.current_image.copy())
        bands = list(self.current_image.split())

        channel_index = {"R": 0, "G": 1, "B": 2, "A": 3}[channel.upper()]

        if channel.upper() == "A" and len(bands) == 3:
            self.ensure_rgba()
            bands = list(self.current_image.split())

        bands[channel_index] = data
        self.current_image = Image.merge(self.current_image.mode, bands)

    def offset_channel(self, channel: str, offset_x: int, offset_y: int = 0) -> None:
        """
        Offset a color channel by a given amount.

        Args:
            channel: 'R', 'G', or 'B'
            offset_x: Horizontal offset in pixels
            offset_y: Vertical offset in pixels
        """
        if channel.upper() not in ["R", "G", "B"]:
            raise ValueError("Channel must be 'R', 'G', or 'B'")

        self.history.append(self.current_image.copy())

        # Get the channel
        channel_data = self.get_channel(channel)

        # Create offset version
        width, height = self.current_image.size
        offset_data = Image.new("L", (width, height), 0)

        # Calculate wrapped coordinates
        for y in range(height):
            for x in range(width):
                src_x = (x - offset_x) % width
                src_y = (y - offset_y) % height
                pixel = channel_data.getpixel((src_x, src_y))

                if isinstance(pixel, tuple):
                    offset_data.putpixel((x, y), pixel)

                else:
                    raise ValueError("getpixel() returned unexpected value")

        # Apply Gaussian blur for smoother transitions
        offset_data = offset_data.filter(ImageFilter.GaussianBlur(0.5))

        # Set the channel
        self.set_channel(channel, offset_data)

    def undo(self) -> bool:
        """
        Undo the last operation.

        Returns:
            bool: True if undo was successful, False if no more history
        """
        if not self.history:
            return False

        self.current_image = self.history.pop()
        return True

    def save(
        self, path: Union[str, Path, BytesIO], format: Optional[str] = None
    ) -> None:
        """
        Save the current image.

        Args:
            path: Output path or BytesIO object
            format: Optional format override (e.g., 'PNG', 'JPEG')
        """
        self.current_image.save(path, format=format)

    def get_current_image(self) -> Image.Image:
        """
        Get the current image state.

        Returns:
            Image.Image: Current image
        """
        return self.current_image.copy()

    def reset(self) -> None:
        """
        Reset to original image.
        """
        self.history = []
        self.current_image = self.original_image.copy()


'''
class ImageProcessor(BaseImageProcessor):
    def __init__(
        self,
        image_input: Union[str, bytes, Image.Image, BytesIO, None],
        points: bool = False,
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
    '''
