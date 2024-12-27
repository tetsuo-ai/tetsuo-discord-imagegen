import argparse
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from ..config.config import ConfigManager


@dataclass
class ParsedCommand:
    command: str
    image_path: Optional[str]
    effects: List[Tuple[str, Dict]]
    animation_params: Dict
    ascii_params: Dict
    output_params: Dict
    preset_name: Optional[str]
    tags: List[str]


class CommandParser:
    """
    Parses and validates user commands for image processing using argparse.
    """

    def __init__(self, configure):
        """
        Initialize command parser with configuration.

        Args:
            config: ConfigManager instance for validation
        """
        self.config = configure
        self.logger = logging.getLogger("CommandParser")

    def _create_parser(self) -> argparse.ArgumentParser:
        """
        Create the argument parser with all supported arguments.

        Returns:
            argparse.ArgumentParser: Configured parser
        """
        parser = argparse.ArgumentParser(description="Image processing command parser")

        # Command subparsers
        subparsers = parser.add_subparsers(dest="command", help="Command to execute")

        # Process command
        process_parser = subparsers.add_parser(
            "process", help="Process image with effects"
        )
        self._add_common_arguments(process_parser)

        # Animate command
        animate_parser = subparsers.add_parser("animate", help="Create animation")
        self._add_common_arguments(animate_parser)
        self._add_animation_arguments(animate_parser)

        # ASCII command
        ascii_parser = subparsers.add_parser("ascii", help="Generate ASCII art")
        self._add_common_arguments(ascii_parser)
        self._add_ascii_arguments(ascii_parser)

        print("finished creating parser")

        return parser

    def _add_common_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add arguments common to all commands."""
        # Input image
        parser.add_argument("image", nargs="?", help="Input image path")
        parser.add_argument(
            "--random", action="store_true", help="Use random image from library"
        )

        # Effects
        for effect, params in self.config.effect_params.items():
            if any("intensity" in p for p in params.values()):
                parser.add_argument(
                    f"--{effect}",
                    nargs="+",
                    type=float,
                    help=f"Apply {effect} effect with intensity",
                )
            else:
                parser.add_argument(
                    f"--{effect}",
                    nargs="+",
                    type=float,
                    help=f"Apply {effect} effect with values",
                )

        # Preset
        parser.add_argument("--preset", help="Use predefined effect combination")

        # Output parameters
        parser.add_argument(
            "--format", choices=["PNG", "JPEG", "GIF"], help="Output format"
        )
        parser.add_argument(
            "--quality", type=int, choices=range(101), help="Output quality (0-100)"
        )

        # Alpha parameters
        for alpha_param in ["alpha", "coloralpha", "rgbalpha"]:
            parser.add_argument(
                f"--{alpha_param}",
                type=int,
                choices=range(256),
                help=f"Set {alpha_param} value (0-255)",
            )

        # Tags
        parser.add_argument("--tags", nargs="*", help="Add tags to output")

    def _add_animation_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add animation-specific arguments."""
        parser.add_argument(
            "--frames",
            type=int,
            default=self.config.animation.default_frames,
            help=f"Number of frames ({self.config.animation.min_frames}-{self.config.animation.max_frames})",
        )
        parser.add_argument(
            "--fps",
            type=int,
            default=self.config.animation.default_fps,
            help=f"Frames per second ({self.config.animation.min_fps}-{self.config.animation.max_fps})",
        )

    def _add_ascii_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add ASCII-specific arguments."""
        parser.add_argument(
            "--cols",
            type=float,
            default=self.config.ascii.default_cols,
            help=f"Number of columns (1-{self.config.ascii.max_cols})",
        )
        parser.add_argument(
            "--scale",
            type=float,
            default=self.config.ascii.default_scale,
            help="Scale factor (0.1-2.0)",
        )

    async def parse_command(
        self, ctx, command_str: str, image_input: Optional[Union[str, Path]] = None
    ) -> ParsedCommand:

        parser = self._create_parser()

        await ctx.send("_create_parser() isn't broken.")

        # Split the command string and parse arguments
        args = parser.parse_args(command_str.split())

        # Initialize result structure
        result = ParsedCommand(
            command=args.command or "process",  # Default to process
            image_path=str(image_input) if image_input else args.image,
            effects=[],
            animation_params={},
            ascii_params={},
            output_params={},
            preset_name=args.preset,
            tags=args.tags or [],
        )

        # Process preset if specified
        if args.preset:
            try:
                preset = self.config.get_preset(args.preset)
                result.effects.extend(
                    [(effect, params) for effect, params in preset["params"].items()]
                )
            except KeyError:
                raise ValueError(f"Unknown preset: {args.preset}")

        if not isinstance(self.config.effect_params, (dict, list)):
            print(
                f"Unexpected type for effect_params: {type(self.config.effect_params)}"
            )
            # Process effects
        for effect in self.config.effect_params:
            if hasattr(args, effect) and getattr(args, effect) is not None:
                values = getattr(args, effect)
                if len(values) == 1:
                    params = (
                        {"intensity": values[0]}
                        if "intensity" in self.config.effect_params[effect]
                        else {"value": values[0]}
                    )
                else:
                    params = (
                        {"intensity": tuple(values[:2])}
                        if "intensity" in self.config.effect_params[effect]
                        else {"values": values}
                    )

                self.config.validate_params(effect, params)
                result.effects.append((effect, params))

        # Process animation parameters
        if args.command == "animate":
            result.animation_params["frames"] = args.frames
            result.animation_params["fps"] = args.fps

            # Validate animation parameters
            if not (
                self.config.animation.min_frames
                <= args.frames
                <= self.config.animation.max_frames
            ):
                raise ValueError(
                    f"Frames must be between {self.config.animation.min_frames} and {self.config.animation.max_frames}"
                )
            if not (
                self.config.animation.min_fps
                <= args.fps
                <= self.config.animation.max_fps
            ):
                raise ValueError(
                    f"FPS must be between {self.config.animation.min_fps} and {self.config.animation.max_fps}"
                )

        # Process ASCII parameters
        if args.command == "ascii":
            result.ascii_params["cols"] = args.cols
            result.ascii_params["scale"] = args.scale

            # Validate ASCII parameters
            if not (0 < args.cols <= self.config.ascii.max_cols):
                raise ValueError(
                    f"Columns must be between 1 and {self.config.ascii.max_cols}"
                )
            if not (0 < args.scale <= 2.0):
                raise ValueError("Scale must be between 0 and 2.0")

        # Process output parameters
        if args.format:
            result.output_params["format"] = args.format
        if args.quality is not None:
            result.output_params["quality"] = args.quality

        # Process alpha parameters
        for param in ["alpha", "coloralpha", "rgbalpha"]:
            if hasattr(args, param) and getattr(args, param) is not None:
                result.output_params[param] = getattr(args, param)

        # Handle random flag
        if args.random:
            images = list(Path(self.config.IMAGES_FOLDER).glob("*.*"))
            if not images:
                raise ValueError(
                    f"No images found int \
                                 {self.config.IMAGES_FOLDER}"
                )
            result.image_path = str(random.choice(images))
            print("Image selected: ", result.image_path)

        # Validate final command
        self._validate_command(result)

        return result

    def _validate_command(self, parsed: ParsedCommand) -> None:
        """
        Validate parsed command for consistency.

        Args:
            parsed: ParsedCommand object to validate

        Raises:
            ValueError: If command is invalid
        """
        # Check for required image input
        if (
            not parsed.image_path
            and not parsed.preset_name
            and not any("--random" in effect for effect in parsed.effects)
        ):
            raise ValueError("No image input specified")

    def format_help(self) -> str:
        """
        Generate help text for available commands.

        Returns:
            str: Formatted help text
        """
        parser = self._create_parser()
        return parser.format_help()

    def get_example_commands(self) -> List[str]:
        """
        Get list of example commands.

        Returns:
            List[str]: Example commands
        """
        return [
            "process --glitch 0.5 --chroma 0.3 --tags cyberpunk",
            "animate --preset psychic --frames 30 --fps 24",
            "ascii --cols 120 --scale 0.5",
            "process --random --preset cyberpunk",
            "animate --glitch 0.3 0.8 --chroma 0.2 0.4",
        ]
