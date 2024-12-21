import asyncio
import logging
import os
import shutil
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tetimi import EFFECT_ORDER, ImageProcessor


class ASCIIAnimationProcessor:
    def __init__(
        self,
        image_input: Union[str, bytes, Image.Image, BytesIO],
        output_dir: str = "animations",
    ):
        """Initialize ASCII animation processor

        Args:
            image_input: Input image in various formats
            output_dir: Directory for output files
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ASCIIAnimationProcessor")

        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / "frames"

        if self.frames_dir.exists():
            shutil.rmtree(self.frames_dir)

        self.output_dir.mkdir(exist_ok=True)
        self.frames_dir.mkdir(exist_ok=True)

        if isinstance(image_input, str):
            self.input_path = Path(image_input)
            if not self.input_path.exists():
                raise ValueError(f"Input image not found: {image_input}")
            self.base_image = Image.open(self.input_path)
            self.temp_input = False
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            self.input_path = Path(tmp.name)
            self.temp_input = True

            if isinstance(image_input, bytes):
                self.base_image = Image.open(BytesIO(image_input))
            elif isinstance(image_input, Image.Image):
                self.base_image = image_input
            elif isinstance(image_input, BytesIO):
                self.base_image = Image.open(image_input)
            else:
                raise ValueError("Unsupported image input type")

            self.base_image.save(tmp.name, format="PNG")

        self.base_processor = ImageProcessor(self.base_image)
        self.executor = ThreadPoolExecutor(max_workers=4)

    def create_frame_image(
        self, ascii_frame: List[str], font_size: int = 14
    ) -> Image.Image:
        """Convert ASCII frame to image

        Args:
            ascii_frame: List of ASCII art lines
            font_size: Font size for rendering

        Returns:
            PIL Image of rendered ASCII art
        """
        try:
            font = ImageFont.truetype("Courier", font_size)
        except:
            font = ImageFont.load_default()

        char_width = font_size * 0.6
        char_height = font_size * 1.2
        max_line_length = max(len(line) for line in ascii_frame)
        width = int(max_line_length * char_width)
        height = int(len(ascii_frame) * char_height)

        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)

        y = 0
        for line in ascii_frame:
            draw.text((0, y), line, font=font, fill="black")
            y += char_height

        return image

    async def generate_frame(
        self,
        frame_num: int,
        total_frames: int,
        effects: Dict[str, Any],
        cols: int,
        scale: float,
    ) -> Path:
        """Generate a single frame asynchronously"""
        progress = frame_num / (total_frames - 1)

        # Calculate frame effects
        frame_effects = {}
        for effect, value in effects.items():
            if isinstance(value, tuple) and len(value) == 2:
                start, end = value
                if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                    frame_effects[effect] = start + (end - start) * progress
            else:
                frame_effects[effect] = value

        # Process image with effects
        processed_image = self.base_image.copy()
        for effect in EFFECT_ORDER:
            if effect in frame_effects:
                processor = ImageProcessor(processed_image)
                processed_image = processor.apply_effect(
                    effect, {effect: frame_effects[effect]}
                )

        # Convert to ASCII
        processor = ImageProcessor(processed_image)

        if "impact" in frame_effects.keys():
            processor.base_image = ImageProcessor(processed_image).apply_impact_text(
                frame_effects["impact"]
            )
        ascii_frame = await asyncio.get_event_loop().run_in_executor(
            self.executor, processor.convertImageToAscii, cols, scale
        )

        # Create and save frame
        frame_image = await asyncio.get_event_loop().run_in_executor(
            self.executor, self.create_frame_image, ascii_frame
        )

        frame_path = self.frames_dir / f"frame_{frame_num:04d}.png"
        await asyncio.get_event_loop().run_in_executor(
            self.executor, frame_image.save, frame_path
        )

        self.logger.info(f"Generated frame {frame_num + 1}/{total_frames}")
        return frame_path

    async def generate_frames(
        self,
        params: Optional[Dict[str, Any]] = None,
        num_frames: int = 30,
        cols: int = 80,
    ) -> List[Path]:
        """Generate ASCII animation frames asynchronously

        Args:
            params: Animation parameters
            num_frames: Number of frames to generate
            cols: Number of columns for ASCII art

        Returns:
            List of frame file paths
        """
        params = params or {}
        scale = params.get("scale", 0.43)

        effects = params.get(
            "effects",
            {"consciousness": (0.3, 0.8), "glitch": (0, 5), "chroma": (5, 15)},
        )

        if "impact" in params.keys():
            effects["impact"] = params["impact"]

        tasks = []
        for i in range(num_frames):
            task = self.generate_frame(i, num_frames, effects, cols, scale)
            tasks.append(task)

        frame_paths = await asyncio.gather(*tasks)

        # Save complete ASCII animation to text file
        frames_text = []
        for i, frame_path in enumerate(frame_paths):
            processor = ImageProcessor(Image.open(frame_path))
            ascii_frame = processor.convertImageToAscii(cols=cols, scale=scale)
            frames_text.append(
                f"=== Frame {i:04d} ===\n" + "\n".join(ascii_frame) + "\n\n"
            )

        text_path = self.output_dir / "ascii_animation.txt"
        with open(text_path, "w", encoding="utf-8") as f:
            f.writelines(frames_text)

        return frame_paths

    def cleanup(self):
        """Clean up temporary files"""
        self.executor.shutdown(wait=False)
        try:
            if hasattr(self, "temp_input") and self.temp_input:
                os.unlink(self.input_path)
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

        if self.frames_dir.exists():
            try:
                shutil.rmtree(self.frames_dir)
            except Exception as e:
                self.logger.error(f"Failed to remove frames directory: {e}")

    def __del__(self):
        """Cleanup on object destruction"""
        self.cleanup()
