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

        """
        # Get the channel
        channel_data = self.get_channel(channel)

        # Create offset version
        width, height = self.current_image.size
        offset_data = Image.new("RGB", (width, height), 0)

        # Calculate wrapped coordinates
        for y in range(height):
            for x in range(width):
                src_x = (x - offset_x) % width
                src_y = (y - offset_y) % height
                pixel = channel_data.getpixel((src_x, src_y))

                if isinstance(pixel, tuple):
                    offset_data.putpixel((x, y), pixel)

                else:
                    continue

        # Apply Gaussian blur for smoother transitions
        offset_data = offset_data.filter(ImageFilter.GaussianBlur(0.5))

        # Set the channel
        self.set_channel(channel, offset_data)
        """
        """Enhanced channel offset with smoother transitions"""
        image = self.current_image
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
        self.current_image = offset_image.resize((width, height), Image.Resampling.LANCZOS)
    

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
        self, path: Union[str, Path, BytesIO], format: Optional[str] = None,
            quality: Optional[int] = 95
    ) -> None:
        """
        Save the current image.

        Args:
            path: Output path or BytesIO object
            format: Optional format override (e.g., 'PNG', 'JPEG')
        """
        self.current_image.save(path, format=format, quality=quality)

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
