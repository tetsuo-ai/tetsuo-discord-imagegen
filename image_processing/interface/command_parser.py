import argparse
import logging
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union, Any

from ..config.config import ConfigManager


class OutputFormat(Enum):
    PNG = auto()
    JPEG = auto()
    GIF = auto()


@dataclass
class OutputParams:
    format: Optional[OutputFormat] = None
    quality: Optional[int] = None
    alpha: Optional[int] = None
    coloralpha: Optional[int] = None
    rgbalpha: Optional[int] = None


@dataclass
class AnimationParams:
    frames: int
    fps: int
    video_crf: int
    video_preset: str
    gif_duration: int


@dataclass
class ASCIIParams:
    cols: int
    scale: float
    font_size: int
    char_set: str


@dataclass
class ParsedCommand:
    command: Literal["process", "animate", "ascii", "image"]
    image_path: Optional[str]
    effects: List[Tuple[str, Dict[str, Union[float, Tuple[float, float]]]]] = field(
        default_factory=list
    )
    animation_params: Optional[AnimationParams] = None
    ascii_params: Optional[ASCIIParams] = None
    output_params: OutputParams = field(default_factory=OutputParams)
    preset_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)


class CommandParser:
    """
    Parser for image processing commands using argparse.
    """

    def __init__(self, configure: ConfigManager):
        self.config = configure
        self.logger = logging.getLogger("CommandParser")

    def _create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Image processing command parser")
        subparsers = parser.add_subparsers(dest="command", help="Command to execute")

        process_parser = subparsers.add_parser(
            "process", help="Process image with effects"
        )
        animate_parser = subparsers.add_parser("animate", help="Create animation")
        ascii_parser = subparsers.add_parser("ascii", help="Generate ASCII art")
        image_parser = subparsers.add_parser("image", help="Input image path")
        self._add_arguments(process_parser)
        self._add_arguments(animate_parser)
        self._add_arguments(ascii_parser)

#        self._add_animation_arguments(animate_parser)
        self._add_ascii_arguments(ascii_parser)

        return parser


    def _build_help(self, option_def: Dict, effect_params: Dict, effect,
                    description: str) -> str:
        help_output = f"--{effect}\n"
        for name in option_def['names']:
            opts = effect_params[effect]['constraints'][name]

            help_output += f"\t[{opts['min']} - {opts['max']}] [default: {opts['default']}]\n"
        help_output += f"{description}\n"

        return help_output


    def _build_option_list(self, constraints: Dict) -> Dict:
        parsed_options = {
            "names": [],
            "types": set(),
            "defaults": [],
        }

        for constraint, key in constraints.items():
            parsed_options['names'].append(constraint)
            parsed_options['types'].add(key['type'])
            parsed_options['defaults'].append(key['default'])

        return parsed_options

    # This needs to be replaced with Dict["command", function]

    def multi_type(self, arg):
        # Assuming args is a list of strings from nargs='+'
        result = ""
        try:
            if arg.index('.') > 0:
                result = float(arg)
        except ValueError:
            try:
                result = int(arg)
            except ValueError:
                result = arg
        return result

    def _add_arguments(self, parser: argparse.ArgumentParser) -> None:
        try:
            effect_dict: Dict[str, Dict[str, Any]] = self.config.effect_params
            effect = str()
            params: Dict[str, Any] = {}

            option_def = {}

            for effect, params in effect_dict.items():
                option_def = self._build_option_list(
                        self.config.effect_params[effect]["constraints"])

            help_msg = self._build_help(option_def, self.config.effect_params,
                                effect, params['description'])
        except Exception as e:
            raise ValueError(f"Error while parsing command: {str(e)}")

        constraints = self.config.effect_params[effect]["constraints"]

        if len(option_def['names']) == 1:
            parser.add_argument(
                f"--{effect}",
                type=constraints[option_def['names'][0]]["type"],
                help=help_msg
            )
        else:
            if len(option_def['types']) == 1:
                parser.add_argument(
                    f"--{effect}",
                    type=next(iter(option_def['types'])),
                    nargs="*",
                    help=help_msg
                )
            else:
                parser.add_argument(
                    f"--{effect}",
                    type=self.multi_type,
                    nargs="*",
                    help=help_msg
                )

    def _add_ascii_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--cols",
            type=int,
            default=self.config.ascii.default_cols,
            help=f"Number of columns (1-{self.config.ascii.max_cols})",
        )
        parser.add_argument(
            "--scale",
            type=float,
            default=self.config.ascii.default_scale,
            help="Scale factor (0.1-2.0)",
        )
        parser.add_argument(
            "--font-size",
            type=int,
            default=self.config.ascii.default_font_size,
            help="Font size for ASCII art",
        )
        parser.add_argument(
            "--detailed",
            action="store_true",
            help="Use detailed character set for ASCII art",
        )

    async def parse_command(
        self, ctx, command_str: str, image_input: Optional[Union[str, Path]] = None
    ) -> ParsedCommand:
        await ctx.send("Got to parse_command()")
        parser = self._create_parser()
        await ctx.send("_create_parser() isn't broken.")

        try:
            args = parser.parse_args(command_str.split())
        except argparse.ArgumentError as e:
            raise ValueError(f"Invalid command arguments: {str(e)}")

        if args.command not in ("process", "animate", "ascii"):
            raise ValueError(f"Invalid command: {args.command}")

        result = ParsedCommand(
            command=args.command,
            image_path=str(image_input) if image_input else args.image,
            preset_name=args.preset,
            tags=args.tags or [],
        )

        if args.preset:
            preset = self.config.get_preset(args.preset)
            if not preset or not isinstance(preset, dict) or "params" not in preset:
                raise ValueError(f"Invalid preset configuration: {args.preset}")
            result.effects.extend(
                [(effect, params) for effect, params in preset["params"].items()]
            )

        for effect, params in self.config.effect_params.items():
            effect_value = getattr(args, effect, None)
            if effect_value is not None:
                params = self._create_effect_params(effect, effect_value, params)
                result.effects.append((effect, params))

        if args.command == "animate":
            result.animation_params = self._create_animation_params(args)
        elif args.command == "ascii":
            result.ascii_params = self._create_ascii_params(args)

        result.output_params = self._create_output_params(args)

        if args.random:
            result.image_path = self._get_random_image_path()

        self._validate_command(result)
        return result

    def _create_effect_params(
        self, effect: str, values: List[float], effect_config: Dict
    ) -> Dict[str, Union[float, Tuple[float, float]]]:
        if not values:
            raise ValueError(f"No values provided for effect: {effect}")

        has_intensity = any("intensity" in p for p in effect_config.values())
        if len(values) == 1:
            return {"intensity": values[0]} if has_intensity else {"value": values[0]}
        return {"intensity": tuple(values[:2])} if has_intensity else {"values": values}

    def _create_animation_params(self, args: argparse.Namespace) -> AnimationParams:
        if not (
            self.config.animation.min_frames
            <= args.frames
            <= self.config.animation.max_frames
        ):
            raise ValueError(
                f"Frames must be between {self.config.animation.min_frames} "
                f"and {self.config.animation.max_frames}"
            )
        if not (
            self.config.animation.min_fps <= args.fps <= self.config.animation.max_fps
        ):
            raise ValueError(
                f"FPS must be between {self.config.animation.min_fps} "
                f"and {self.config.animation.max_fps}"
            )

        return AnimationParams(
            frames=args.frames,
            fps=args.fps,
            video_crf=args.video_crf,
            video_preset=args.video_preset,
            gif_duration=args.gif_duration,
        )

    def _create_ascii_params(self, args: argparse.Namespace) -> ASCIIParams:
        if not (0 < args.cols <= self.config.ascii.max_cols):
            raise ValueError(
                f"Columns must be between 1 and {self.config.ascii.max_cols}"
            )
        if not (0 < args.scale <= 2.0):
            raise ValueError("Scale must be between 0 and 2.0")

        char_set = (
            self.config.ascii.detailed_chars
            if args.detailed
            else self.config.ascii.basic_chars
        )
        return ASCIIParams(
            cols=args.cols,
            scale=args.scale,
            font_size=args.font_size,
            char_set=char_set,
        )

    def _create_output_params(self, args: argparse.Namespace) -> OutputParams:
        output_params = OutputParams()
        if args.format:
            output_params.format = OutputFormat[args.format]
        if args.quality is not None:
            output_params.quality = args.quality
        for param in ("alpha", "coloralpha", "rgbalpha"):
            value = getattr(args, param, None)
            if value is not None:
                setattr(output_params, param, value)
        return output_params

    def _get_random_image_path(self) -> str:
        images = list(Path(self.config.IMAGES_FOLDER).glob("*.*"))
        if not images:
            raise ValueError(f"No images found in {self.config.IMAGES_FOLDER}")
        selected_path = str(random.choice(images))
        print("Image selected:", selected_path)
        return selected_path

    def _validate_command(self, parsed: ParsedCommand) -> None:
        if (
            not parsed.image_path
            and not parsed.preset_name
            and not any("--random" in effect[0] for effect in parsed.effects)
        ):
            raise ValueError("No image input specified")
