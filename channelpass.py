import logging
import os
import shutil
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from tetimi import ImageProcessor


class ChannelPassAnimator:
    def __init__(
        self,
        image_input: Union[str, bytes, Image.Image, BytesIO],
        output_dir: str = "animations",
    ):
        """Initialize animator with multiple input type support"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ChannelPassAnimator")

        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / "frames"

        # Handle different input types
        if isinstance(image_input, str):
            self.input_path = Path(image_input)
            if not self.input_path.exists():
                raise ValueError(f"Input image not found: {image_input}")
            self.base_image = Image.open(self.input_path)
            self.temp_input = False
        else:
            if isinstance(image_input, bytes):
                self.base_image = Image.open(BytesIO(image_input))
            elif isinstance(image_input, Image.Image):
                self.base_image = image_input
            elif isinstance(image_input, BytesIO):
                self.base_image = Image.open(image_input)
            else:
                raise ValueError("Unsupported image input type")
            self.temp_input = True

        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        if self.frames_dir.exists():
            shutil.rmtree(self.frames_dir)
        self.frames_dir.mkdir(exist_ok=True)

        # Store frames for GIF creation
        self.frames = []

    def ensure_even_dimensions(self, image: Image.Image) -> Image.Image:
        """Ensure image dimensions are even"""
        width, height = image.size
        new_width = width if width % 2 == 0 else width + 1
        new_height = height if height % 2 == 0 else height + 1

        if new_width != width or new_height != height:
            new_image = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))
            new_image.paste(
                image, ((new_width - width) // 2, (new_height - height) // 2)
            )
            return new_image
        return image

    def interpolate_keyframes(
        self, keyframes: List[float], total_frames: int
    ) -> List[float]:
        """Interpolate between keyframe values"""
        if len(keyframes) < 2:
            return [keyframes[0]] * total_frames

        segments = len(keyframes) - 1
        frames_per_segment = total_frames // segments

        interpolated = []
        for i in range(segments):
            start_val = keyframes[i]
            end_val = keyframes[i + 1]

            # Calculate frames for this segment
            if i == segments - 1:
                segment_frames = total_frames - len(interpolated)
            else:
                segment_frames = frames_per_segment

            for frame in range(segment_frames):
                progress = frame / segment_frames
                value = start_val + (end_val - start_val) * progress
                interpolated.append(value)

        return interpolated

    def create_channel_pass_frame(
        self, offset_values: Tuple[float, float]
    ) -> Image.Image:
        """Create a frame with RGB channel offsets"""
        img = self.base_image.convert("RGBA")
        width, height = img.size

        r, g, b, a = img.split()

        g_offset = int(offset_values[0] * width) % width
        b_offset = int(offset_values[1] * width) % width

        # Create output image
        result = Image.new("RGBA", (width, height), (0, 0, 0, 0))

        # Create wrapped green channel
        g_temp = Image.new("L", (width, height))
        g_temp.paste(g.crop((0, 0, width - g_offset, height)), (g_offset, 0))
        g_temp.paste(g.crop((width - g_offset, 0, width, height)), (0, 0))

        # Create wrapped blue channel
        b_temp = Image.new("L", (width, height))
        b_temp.paste(b.crop((0, 0, width - b_offset, height)), (b_offset, 0))
        b_temp.paste(b.crop((width - b_offset, 0, width, height)), (0, 0))

        # Merge channels back together
        result = Image.merge("RGBA", (r, g_temp, b_temp, a))
        return result

    def generate_frames(
        self, params: Dict[str, Any], num_frames: int = 60
    ) -> Optional[Path]:
        """Generate frames with keyframe interpolation and create GIF"""
        try:
            self.base_image = self.ensure_even_dimensions(self.base_image)
            self.frames = []  # Reset frames list

            # Get keyframe values
            g_keyframes = params.get("g_values", [0, 0.2, 0])
            b_keyframes = params.get("b_values", [0, 0.3, 0])

            # Interpolate keyframe values
            g_values = self.interpolate_keyframes(g_keyframes, num_frames)
            b_values = self.interpolate_keyframes(b_keyframes, num_frames)

            impact_text = ""

            if "impact" in params.keys():
                impact_text = params["impact"]

            for i in range(num_frames):
                # Create frame with current offset values
                frame = self.create_channel_pass_frame((g_values[i], b_values[i]))

                # Resize if image is too large
                if frame.size[0] > 800 or frame.size[1] > 800:
                    aspect_ratio = frame.size[0] / frame.size[1]
                    if aspect_ratio > 1:
                        new_size = (800, int(800 / aspect_ratio))
                    else:
                        new_size = (int(800 * aspect_ratio), 800)
                    frame = frame.resize(new_size, Image.Resampling.LANCZOS)

                if impact_text != "":
                    ip = ImageProcessor(frame)
                    frame = ip.apply_impact_text(impact_text)

                self.frames.append(frame)  # Store frame for GIF

                # Save individual frame for reference
                frame_path = self.frames_dir / f"frame_{i:04d}.png"
                frame.save(frame_path)

                self.logger.info(f"Generated frame {i + 1}/{num_frames}")

            # Create and return GIF
            return self.create_gif()

        except Exception as e:
            self.logger.error(f"Error generating frames: {str(e)}")
            raise

    def create_gif(self, duration: int = 50) -> Optional[Path]:
        """Create GIF from generated frames"""
        if not self.frames:
            raise ValueError("No frames available to create GIF")

        try:
            gif_path = self.output_dir / "animation.gif"

            # Optimize frames for GIF
            optimized_frames = []
            for frame in self.frames:
                # Convert to P mode with adaptive palette
                if frame.mode != "P":
                    frame = frame.convert("RGBA").convert(
                        "P", palette=Image.Palette.ADAPTIVE, colors=256
                    )
                optimized_frames.append(frame)

            # Save as GIF
            optimized_frames[0].save(
                gif_path,
                save_all=True,
                append_images=optimized_frames[1:],
                duration=duration,  # Duration between frames in milliseconds
                loop=0,  # 0 means loop forever
                optimize=True,
            )

            self.logger.info(f"Created GIF: {gif_path}")
            return gif_path

        except Exception as e:
            self.logger.error(f"Error creating GIF: {str(e)}")
            raise

    def cleanup(self):
        """Remove temporary files"""
        if self.frames_dir.exists():
            shutil.rmtree(self.frames_dir)
        self.logger.info("Cleaned up temporary files")

    def __del__(self):
        """Cleanup on object destruction"""
        self.cleanup()
