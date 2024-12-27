import logging
import os
import shutil
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from ..core.effect_processor import EffectProcessor
from ..core.image_processor import BaseImageProcessor
from ..core.utils import ImageUtils


class AnimationProcessor:
    """
    Handles creation and management of image effect animations with improved multi-effect support.
    """

    def __init__(self, image_input: Union[str, bytes, Image.Image, BytesIO]):
        """Initialize animation processor with enhanced effect handling."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("AnimationProcessor")

        # Load and validate input image
        self.base_image = ImageUtils.load_image(image_input)

        # Ensure dimensions are even for video encoding
        width, height = self.base_image.size
        new_width, new_height = ImageUtils.ensure_size_even(width, height)

        if (new_width, new_height) != (width, height):
            new_image = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))
            new_image.paste(
                self.base_image, ((new_width - width) // 2, (new_height - height) // 2)
            )
            self.base_image = new_image

        # Set up temporary directory for frames
        self.temp_dir = Path(tempfile.mkdtemp(prefix="anim_frames_"))
        self.frames_dir = self.temp_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)

    def _validate_effects(self, effects: List[Tuple[str, Dict[str, Any]]]) -> None:
        """Validate effect parameters before processing."""
        valid_effects = {
            "glitch",
            "chroma",
            "scan",
            "noise",
            "energy",
            "pulse",
            "consciousness",
        }

        for effect_name, params in effects:
            if effect_name not in valid_effects:
                raise ValueError(f"Invalid effect: {effect_name}")

            # Normalize intensity parameters
            if "intensity" in params:
                intensity = params["intensity"]
                if isinstance(intensity, (int, float)):
                    params["intensity"] = min(1.0, max(0.0, float(intensity) / 100))
                elif isinstance(intensity, tuple):
                    params["intensity"] = (
                        min(1.0, max(0.0, float(intensity[0]) / 100)),
                        min(1.0, max(0.0, float(intensity[1]) / 100)),
                    )

    def _interpolate_parameters(
        self, params: Dict[str, Any], progress: float
    ) -> Dict[str, Any]:
        """Interpolate effect parameters for current frame."""
        frame_params = {}
        for param_name, param_value in params.items():
            if isinstance(param_value, tuple) and len(param_value) == 2:
                frame_params[param_name] = ImageUtils.interpolate_value(
                    param_value[0], param_value[1], progress
                )
            else:
                frame_params[param_name] = param_value
        return frame_params

    def generate_frames(
        self, effects: List[Tuple[str, Dict[str, Any]]], num_frames: int = 30
    ) -> List[Path]:
        """Generate animation frames with improved multi-effect support."""
        frame_paths = []
        self._validate_effects(effects)

        try:
            for i in range(num_frames):
                progress = i / (num_frames - 1)

                # Start with fresh copy of base image for each frame
                frame = self.base_image.copy()
                processor = EffectProcessor(frame)

                # Apply effects in sequence with proper parameter interpolation
                for effect_name, params in effects:
                    frame_params = self._interpolate_parameters(params, progress)

                    try:
                        processor.apply_effect(effect_name, frame_params)
                    except Exception as e:
                        self.logger.warning(
                            f"Error applying effect {effect_name}: {str(e)}"
                        )
                        continue

                # Save frame
                frame_path = self.frames_dir / f"frame_{i:04d}.png"
                processor.save(frame_path)
                frame_paths.append(frame_path)

                self.logger.info(f"Generated frame {i + 1}/{num_frames}")

        except Exception as e:
            self.logger.error(f"Frame generation error: {str(e)}")
            raise

        return frame_paths

    def create_video(
        self,
        frame_paths: List[Path],
        output_path: Optional[Union[str, Path]] = None,
        frame_rate: int = 24,
        crf: int = 23,
        preset: str = "medium",
    ) -> Optional[Path]:
        """Create video with improved error handling and frame management."""
        if not frame_paths:
            raise ValueError("No frames provided for video creation")

        if not output_path:
            output_path = self.temp_dir / "output.mp4"
        else:
            output_path = Path(output_path)

        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Verify all frames exist
            for frame_path in frame_paths:
                if not frame_path.exists():
                    raise FileNotFoundError(f"Missing frame: {frame_path}")

            # Construct ffmpeg command
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(frame_rate),
                "-i",
                str(self.frames_dir / "frame_%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                str(crf),
                "-preset",
                preset,
                "-movflags",
                "+faststart",
                "-vf",
                "format=yuv420p",
                str(output_path),
            ]

            # Run ffmpeg with proper error handling
            try:
                result = subprocess.run(
                    ffmpeg_cmd, check=True, capture_output=True, text=True
                )

                if result.returncode == 0:
                    return output_path

            except subprocess.CalledProcessError as e:
                self.logger.error(f"ffmpeg error: {e.stderr}")
                raise

        except Exception as e:
            self.logger.error(f"Video creation error: {str(e)}")
            raise

        return None

    def cleanup(self):
        """Clean up temporary files with improved error handling."""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")

    def __del__(self):
        """Ensure cleanup on object destruction."""
        self.cleanup()


class ASCIIProcessor:
    """
    Handles creation and management of image effect animations.
    """

    def __init__(self, image_input: Union[str, bytes, Image.Image, BytesIO]):
        """
        Initialize animation processor.

        Args:
            image_input: Source image in various formats
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("AnimationProcessor")

        # Load and validate input image
        self.base_image = ImageUtils.load_image(image_input)

        # Ensure dimensions are even for video encoding
        width, height = self.base_image.size
        new_width, new_height = ImageUtils.ensure_size_even(width, height)

        if (new_width, new_height) != (width, height):
            new_image = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))
            new_image.paste(
                self.base_image, ((new_width - width) // 2, (new_height - height) // 2)
            )
            self.base_image = new_image

        # Set up temporary directory for frames
        self.temp_dir = Path(tempfile.mkdtemp(prefix="anim_frames_"))
        self.frames_dir = self.temp_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)

        # Initialize processors
        self.effect_processor = EffectProcessor(self.base_image)

    def convert_to_ascii(
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

    def generate_frames(
        self, effects: List[Tuple[str, Dict[str, Any]]], num_frames: int = 30
    ) -> List[Path]:
        """
        Generate animation frames with multiple effects.

        Args:
            effects: List of (effect_name, parameters) tuples
            num_frames: Number of frames to generate

        Returns:
            List of paths to generated frame files
        """
        frame_paths = []

        try:
            for i in range(num_frames):
                # Calculate animation progress
                progress = i / (num_frames - 1)

                # Process frame with interpolated parameters
                frame = self.base_image.copy()
                processor = EffectProcessor(frame)

                for effect_name, params in effects:
                    # Interpolate parameters
                    frame_params = {}
                    for param_name, param_value in params.items():
                        if isinstance(param_value, tuple) and len(param_value) == 2:
                            frame_params[param_name] = ImageUtils.interpolate_value(
                                param_value[0], param_value[1], progress
                            )
                        else:
                            frame_params[param_name] = param_value

                    # Apply effect
                    processor.apply_effect(effect_name, frame_params)

                # Save frame
                frame_path = self.frames_dir / f"frame_{i:04d}.png"
                processor.save(frame_path)
                frame_paths.append(frame_path)

                self.logger.info(f"Generated frame {i + 1}/{num_frames}")

        except Exception as e:
            self.logger.error(f"Frame generation error: {str(e)}")
            raise

        return frame_paths

    def create_video(
        self,
        frame_paths: List[Path],
        output_path: Optional[Union[str, Path]] = None,
        frame_rate: int = 24,
        crf: int = 23,
        preset: str = "medium",
    ) -> Optional[Path]:
        """
        Create video from frames using ffmpeg.

        Args:
            frame_paths: List of frame file paths
            output_path: Path for output video file
            frame_rate: Frames per second
            crf: Constant Rate Factor (18-28 recommended)
            preset: ffmpeg encoding preset

        Returns:
            Path to output video file
        """
        if not frame_paths:
            raise ValueError("No frames provided for video creation")

        if not output_path:
            output_path = self.temp_dir / "output.mp4"
        else:
            output_path = Path(output_path)

        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Construct ffmpeg command
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(frame_rate),
                "-i",
                str(self.frames_dir / "frame_%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                str(crf),
                "-preset",
                preset,
                "-movflags",
                "+faststart",
                "-vf",
                "format=yuv420p",
                str(output_path),
            ]

            # Run ffmpeg
            result = subprocess.run(
                ffmpeg_cmd, check=True, capture_output=True, text=True
            )

            if result.returncode == 0:
                return output_path

        except subprocess.CalledProcessError as e:
            self.logger.error(f"ffmpeg error: {e.stderr}")
        except Exception as e:
            self.logger.error(f"Video creation error: {str(e)}")

        return None

    def create_gif(
        self,
        frame_paths: List[Path],
        output_path: Optional[Union[str, Path]] = None,
        duration: int = 50,
    ) -> Optional[Path]:
        """
        Create animated GIF from frames.

        Args:
            frame_paths: List of frame file paths
            output_path: Path for output GIF file
            duration: Frame duration in milliseconds

        Returns:
            Path to output GIF file
        """
        if not frame_paths:
            raise ValueError("No frames provided for GIF creation")

        if not output_path:
            output_path = self.temp_dir / "output.gif"
        else:
            output_path = Path(output_path)

        try:
            # Load frames and optimize for GIF
            frames = []
            for frame_path in frame_paths:
                with Image.open(frame_path) as frame:
                    # Convert to P mode with adaptive palette
                    if frame.mode != "P":
                        frame = frame.convert("RGBA").convert(
                            "P", palette=Image.Palette.ADAPTIVE, colors=256
                        )
                    frames.append(frame.copy())

            # Save as GIF
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0,
                optimize=True,
            )

            return output_path

        except Exception as e:
            self.logger.error(f"GIF creation error: {str(e)}")
            return None

    def create_ascii_animation(
        self,
        effects: List[Tuple[str, Dict[str, Any]]],
        num_frames: int = 30,
        cols: int = 120,
        scale: float = 0.43,
    ) -> List[str]:
        """
        Create ASCII art animation frames.

        Args:
            effects: List of (effect_name, parameters) tuples
            num_frames: Number of frames to generate
            cols: Number of columns for ASCII art
            scale: Character aspect ratio adjustment

        Returns:
            List of ASCII art strings
        """
        ascii_frames = []

        try:
            for i in range(num_frames):
                progress = i / (num_frames - 1)

                # Process frame with effects
                frame = self.base_image.copy()
                processor = EffectProcessor(frame)

                for effect_name, params in effects:
                    frame_params = {}
                    for param_name, param_value in params.items():
                        if isinstance(param_value, tuple) and len(param_value) == 2:
                            frame_params[param_name] = ImageUtils.interpolate_value(
                                param_value[0], param_value[1], progress
                            )
                        else:
                            frame_params[param_name] = param_value

                    processor.apply_effect(effect_name, frame_params)

                # Convert to ASCII
                ascii_frame = processor.convertImageToAscii(cols=cols, scale=scale)
                ascii_frames.append("\n".join(ascii_frame))

                self.logger.info(f"Generated ASCII frame {i + 1}/{num_frames}")

        except Exception as e:
            self.logger.error(f"ASCII animation error: {str(e)}")
            raise

        return ascii_frames

    def cleanup(self):
        """Clean up temporary files."""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")

    def __del__(self):
        """Ensure cleanup on object destruction."""
        self.cleanup()


'''
class ASCIIAnimationProcessor:
    """
    Handles creation and management of image effect animations.
    """

    def __init__(self, image_input: Union[str, bytes, Image.Image, BytesIO]):
        """
        Initialize animation processor.

        Args:
            image_input: Source image in various formats
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("AnimationProcessor")

        # Load and validate input image
        self.base_image = ImageUtils.load_image(image_input)

        # Ensure dimensions are even for video encoding
        width, height = self.base_image.size
        new_width, new_height = ImageUtils.ensure_size_even(width, height)

        if (new_width, new_height) != (width, height):
            new_image = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))
            new_image.paste(
                self.base_image, ((new_width - width) // 2, (new_height - height) // 2)
            )
            self.base_image = new_image

        # Set up temporary directory for frames
        self.temp_dir = Path(tempfile.mkdtemp(prefix="anim_frames_"))
        self.frames_dir = self.temp_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)

        # Initialize processors
        self.effect_processor = EffectProcessor(self.base_image)

    def generate_frames(
        self, effects: List[Tuple[str, Dict[str, Any]]], num_frames: int = 30
    ) -> List[Path]:
        """
        Generate animation frames with multiple effects.

        Args:
            effects: List of (effect_name, parameters) tuples
            num_frames: Number of frames to generate

        Returns:
            List of paths to generated frame files
        """
        frame_paths = []

        try:
            for i in range(num_frames):
                # Calculate animation progress
                progress = i / (num_frames - 1)

                # Process frame with interpolated parameters
                frame = self.base_image.copy()
                processor = EffectProcessor(frame)

                for effect_name, params in effects:
                    # Interpolate parameters
                    frame_params = {}
                    for param_name, param_value in params.items():
                        if isinstance(param_value, tuple) and len(param_value) == 2:
                            frame_params[param_name] = ImageUtils.interpolate_value(
                                param_value[0], param_value[1], progress
                            )
                        else:
                            frame_params[param_name] = param_value

                    # Apply effect
                    processor.apply_effect(effect_name, frame_params)

                # Save frame
                frame_path = self.frames_dir / f"frame_{i:04d}.png"
                processor.save(frame_path)
                frame_paths.append(frame_path)

                self.logger.info(f"Generated frame {i + 1}/{num_frames}")

        except Exception as e:
            self.logger.error(f"Frame generation error: {str(e)}")
            raise

        return frame_paths

    def create_video(
        self,
        frame_paths: List[Path],
        output_path: Optional[Union[str, Path]] = None,
        frame_rate: int = 24,
        crf: int = 23,
        preset: str = "medium",
    ) -> Optional[Path]:
        """
        Create video from frames using ffmpeg.

        Args:
            frame_paths: List of frame file paths
            output_path: Path for output video file
            frame_rate: Frames per second
            crf: Constant Rate Factor (18-28 recommended)
            preset: ffmpeg encoding preset

        Returns:
            Path to output video file
        """
        if not frame_paths:
            raise ValueError("No frames provided for video creation")

        if not output_path:
            output_path = self.temp_dir / "output.mp4"
        else:
            output_path = Path(output_path)

        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Construct ffmpeg command
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(frame_rate),
                "-i",
                str(self.frames_dir / "frame_%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                str(crf),
                "-preset",
                preset,
                "-movflags",
                "+faststart",
                "-vf",
                "format=yuv420p",
                str(output_path),
            ]

            # Run ffmpeg
            result = subprocess.run(
                ffmpeg_cmd, check=True, capture_output=True, text=True
            )

            if result.returncode == 0:
                return output_path

        except subprocess.CalledProcessError as e:
            self.logger.error(f"ffmpeg error: {e.stderr}")
        except Exception as e:
            self.logger.error(f"Video creation error: {str(e)}")

        return None

    def create_gif(
        self,
        frame_paths: List[Path],
        output_path: Optional[Union[str, Path]] = None,
        duration: int = 50,
    ) -> Optional[Path]:
        """
        Create animated GIF from frames.

        Args:
            frame_paths: List of frame file paths
            output_path: Path for output GIF file
            duration: Frame duration in milliseconds

        Returns:
            Path to output GIF file
        """
        if not frame_paths:
            raise ValueError("No frames provided for GIF creation")

        if not output_path:
            output_path = self.temp_dir / "output.gif"
        else:
            output_path = Path(output_path)

        try:
            # Load frames and optimize for GIF
            frames = []
            for frame_path in frame_paths:
                with Image.open(frame_path) as frame:
                    # Convert to P mode with adaptive palette
                    if frame.mode != "P":
                        frame = frame.convert("RGBA").convert(
                            "P", palette=Image.Palette.ADAPTIVE, colors=256
                        )
                    frames.append(frame.copy())

            # Save as GIF
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0,
                optimize=True,
            )

            return output_path

        except Exception as e:
            self.logger.error(f"GIF creation error: {str(e)}")
            return None

    def create_ascii_animation(
        self,
        effects: List[Tuple[str, Dict[str, Any]]],
        num_frames: int = 30,
        cols: int = 120,
        scale: float = 0.43,
    ) -> List[str]:
        """
        Create ASCII art animation frames.

        Args:
            effects: List of (effect_name, parameters) tuples
            num_frames: Number of frames to generate
            cols: Number of columns for ASCII art
            scale: Character aspect ratio adjustment

        Returns:
            List of ASCII art strings
        """
        ascii_frames = []

        try:
            for i in range(num_frames):
                progress = i / (num_frames - 1)

                # Process frame with effects
                frame = self.base_image.copy()
                processor = EffectProcessor(frame)

                for effect_name, params in effects:
                    frame_params = {}
                    for param_name, param_value in params.items():
                        if isinstance(param_value, tuple) and len(param_value) == 2:
                            frame_params[param_name] = ImageUtils.interpolate_value(
                                param_value[0], param_value[1], progress
                            )
                        else:
                            frame_params[param_name] = param_value

                    processor.apply_effect(effect_name, frame_params)

                # Convert to ASCII
                ascii_frame = processor.convertImageToAscii(cols=cols, scale=scale)
                ascii_frames.append("\n".join(ascii_frame))

                self.logger.info(f"Generated ASCII frame {i + 1}/{num_frames}")

        except Exception as e:
            self.logger.error(f"ASCII animation error: {str(e)}")
            raise

        return ascii_frames

    def cleanup(self):
        """Clean up temporary files."""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")

    def __del__(self):
        """Ensure cleanup on object destruction."""
        self.cleanup()
'''
