import argparse
import logging
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

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
    command: str
    image_path: Optional[str]
    effects: Dict[str, Dict[str, Any]]
    animation_params: Optional[AnimationParams] = None
    ascii_params: Optional[ASCIIParams] = None
    output_params: OutputParams = field(default_factory=OutputParams)
    tags: List[str] = field(default_factory=list)


class CommandParser:
    """
    Parser for image processing commands using argparse.
    """

    def __init__(self, configure: ConfigManager):
        self.config = configure
        self.logger = logging.getLogger("CommandParser")
        self.effect_params = configure.effect_params
        self._create_parser()
        self.commands = [command for command in self.parser_hook.keys()]

    def _create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
                description="Image processing command parser")
        subparsers = parser.add_subparsers(
                dest="command", help="Command to execute")

        # Initialize Subparsers
        self.animate_parser = subparsers.add_parser("animate", help="Create animation")
        self.ascii_parser = subparsers.add_parser("ascii", help="Generate ASCII art")
        self.image_parser = subparsers.add_parser("image", help="Input image path")
#        self._add_arguments(animate_parser)
        self._add_ascii_arguments(self.ascii_parser)
        self._add_effect_arguments(self.image_parser)

        self.parser_hook = {"animate": self.animate_parser,
                            "ascii": self.ascii_parser,
                            "image": self.image_parser}

        self.command_parsers = subparsers
        return parser

    def _build_help(self, option_def: Dict, effect_params: Dict, effect,
                    description: str) -> str:
        help_output = f"--{effect}\n"
        if option_def['names'][0] is not None:
            if effect != 'random':
                for opt_name in option_def['names']:
                    opts = effect_params[effect]['constraints'][opt_name]

                    help_output += \
                        f"\t[{opts['min']} - {opts['max']}] [default: {opts['default']}]\n"
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
            if not key:
                continue
            if next(iter(key)) is None:
                parsed_options['types'].add("None")
            else:
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

    def _add_effect_arguments(self, parser: argparse.ArgumentParser) -> None:
        try:
            effect_dict: Dict[str, Dict[str, Any]] = self.effect_params
            effect = str()
            params: Dict[str, Any] = {}

            option_def = {}

            for effect, params in effect_dict.items():
                option_def.update(self._build_option_list(
                        self.effect_params[effect]["constraints"]))
                help_msg = self._build_help(option_def, self.effect_params,
                                        effect, params['description'])

                constraints = self.effect_params[effect]["constraints"]

                if effect == 'random':
                    parser.add_argument(
                            f"--{effect}",
                            action="store_true",
                            help=help_msg
                    )
                elif len(option_def['names']) == 1:
                    parser.add_argument(
                        f"--{effect}",
                        type=constraints[option_def['names'][0]]["type"],
                        help=help_msg
                    )
                else:
                    # Handle single option arguments
                    if len(option_def['types']) == 1:
                        parser.add_argument(
                            f"--{effect}",
                            type=next(iter(option_def['types'])),
                            nargs="*",
                            help=help_msg
                        )
                    else:
                        # Handle multiple option arguments
                        parser.add_argument(
                            f"--{effect}",
                            type=self.multi_type,
                            nargs="*",
                            help=help_msg
                        )
        except Exception as e:
            raise ValueError(f"Error while parsing command: {str(e)}")

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

    async def format_help(self, ctx):
        help_out = str()
        for _, subparser in self.command_parsers._name_parser_map.items():
            if subparser is None:
                await ctx.send("No subparser found!")
            for action in subparser._actions:
                if action != 'help':
                    if action.help:
                        help_out += str(action.help)

        await ctx.send(help_out)

    async def parse_command(
        self, ctx, command_str: str, image_input: Optional[Union[str, Path]] = None
    ) -> ParsedCommand:

        arg_list = command_str.split()
        command = arg_list[0]

        try:
            parser = self.parser_hook[command]
        except Exception as e:
            raise ValueError(f"Invalid command: {str(e)}")

        try:
            await ctx.send(f"{arg_list}")
            args = parser.parse_args(arg_list[1:])
            await ctx.send(f"Parser args found: {vars(args)}")
        except argparse.ArgumentError as e:
            raise ValueError(f"Invalid command arguments: {str(e)}")
        except SystemExit as e:
            # Handle SystemExit specifically to prevent it from terminating the bot
            raise ValueError(f"Argument parsing failed: {e}")

        if command not in self.commands:
            raise ValueError(f"Invalid command: {command}")

        try:
            user_effects = {}
            my_args = {k: v for k, v in vars(args).items() if v is not None}
            for effect, option in my_args.items():
                constraints = self.effect_params[effect]["constraints"]
                if isinstance(option, list):
                    constraint_names = list(constraints.keys())
                    user_effects[effect] = {}
                    for i, item in enumerate(option):
                        # Only assign if there are more constraints to match
                        if i < len(constraint_names):
                            user_effects[effect][constraint_names[i]] = item
                else:
                    # Here we check if there's only one constraint for non-list options
                    if len(constraints) == 1:
                        user_effects[effect] = {list(constraints.keys())[0]: option}
                    else:
                        # If there are multiple constraints, you might need a different approach
                        # or perhaps raise an error or log a warning
                        self.logger.warning(f"Multiple constraints for effect {effect} but only one value provided: {option}")
                        user_effects[effect] = {list(constraints.keys())[0]: option}

        except Exception as e:
            raise ValueError(f"Error while parsing command: {str(e)}")
        if "--random" in user_effects:
            image_path = self._get_random_image_path()
        else:
            image_path = \
                    str(image_input) if image_input else self.config.INPUT_IMAGE

        result = ParsedCommand(
            command=command,
            image_path=image_path,
            effects=user_effects,
            tags=[],
        )

        if "animate" in user_effects:
            result.animation_params = self._create_animation_params(args)
        elif command == "ascii":
            result.ascii_params = self._create_ascii_params(args)

        result.output_params = self._create_output_params(args)

        if args.random:
            result.image_path = self._get_random_image_path()

        self._validate_command(result)
        self.current_parser = result
        return result


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
