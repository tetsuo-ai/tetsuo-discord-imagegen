import argparse
import shlex
import logging
import random
from dataclasses import dataclass, fields, MISSING
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from ..config.config import ConfigManager, IMAGES_FOLDER, INPUT_IMAGE


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
    char_set: str = " .,:;iI+#X@"  # An even more robust charset for ASCII art


@dataclass
class ParsedCommand:
    command: str
    image_path: str
    effects: Dict[str, Dict]
    tags: List[str]
    animation_params: Optional[AnimationParams] = None
    ascii_params: Optional[ASCIIParams] = None
    output_params: Optional[Dict] = None


class CommandParser(ConfigManager):
    def __init__(self, config):
        self.effect_params = ConfigManager._build_effect_argument(self)
        self.config = config
        self.current_parser = None
        self.commands = ["image", "ascii", "animate"]
        self._create_parser()
        self.logger = logging.getLogger("CommandParser")

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
        self.add_ascii_arguments(self.ascii_parser)
        self.add_image_arguments(self.image_parser, self.effect_params)

        self.parser_hook = {"animate": self.animate_parser,
                            "ascii": self.ascii_parser,
                            "image": self.image_parser}

        self.command_parsers = subparsers
        return parser


    def add_ascii_arguments(self, subparser):
        for field in fields(ASCIIParams):
            subparser.add_argument(f'--{field.name}', 
                                   type=field.type, 
                                   default=field.default if field.default != MISSING else None,
                                   help=f"Optional: Set {field.name} for ASCII art generation.")

    def add_image_arguments(self, subparser: argparse.ArgumentParser, effect_params):
        for effect, params in effect_params.items():
            if effect == "random":
                subparser.add_argument(
                        "--random",
                        action="store_true",
                        help=f"Use a random image from image repository."
                )
                continue
            group = subparser.add_argument_group(f"{effect} effect")
            if "nargs" in params:
                # If nargs is defined for the effect, use it for all its parameters
                nargs = params["nargs"]
                type_key = next(iter(params["constraints"].keys()))
                my_type = params["constraints"][type_key]["type"]
                group.add_argument(
                    f"--{effect}",
                    type=my_type,
                    nargs=nargs,
                    help=f"{effect_params[effect].get('description', '')}. "
                )
            else:
                # For effects without nargs, add each parameter as a single optional argument
                constraint = next(iter(effect_params[effect]["constraints"].keys()))
                constraint_type = effect_params[effect]["constraints"][constraint]["type"]

                group.add_argument(
                    f"--{effect}",
                    type=constraint_type,
                    help=f"{effect_params[effect].get('description', '')}. ",
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
        command, args = self.parse_command_string(command_str)
        if command_str[0] != "image":
            user_effects = {}
        else:
            user_effects = self.handle_effects(args)
        image_path = self.get_image_path(args, image_input)

        result = ParsedCommand(
            command=command, image_path=image_path, effects=user_effects, tags=[]
        )

        self.add_special_params(result, args)
        self._validate_command(result)  # Assuming this method exists
        self.current_parser = result
        if "random" in user_effects:
            result.image_path = self._get_random_image_path()
        return result

    def parse_command_string(self, command_str: str) -> Tuple[str, argparse.Namespace]:
        arg_list = shlex.split(command_str)
        self.logger.debug(f"arg_list = {arg_list}")
        command = arg_list[0]
        self.logger.debug(f"Command = {command}")

        try:
            parser = self.parser_hook[command]
        except KeyError:
            raise ValueError(f"Invalid command: {command}")

        try:
            args = parser.parse_args(arg_list[1:])
        except argparse.ArgumentError as e:
            raise ValueError(f"Invalid command arguments: {str(e)}")
        except SystemExit as e:
            raise ValueError(f"Argument parsing failed: {e}")

        return command, args

    def handle_effects(self, args: argparse.Namespace) -> Dict[str, Dict]:
        user_effects = {}
        my_args = {k: v for k, v in vars(args).items() if v is not None}
        self.logger.debug(f"my_args: {my_args.items()}")

        for effect, option in my_args.items():
            constraints = self.effect_params.get(effect, {}).get("constraints", {})

            if effect == "impact":
                if isinstance(option, list):
                    user_effects[effect] = {"text": " ".join(option)}
                else:
                    user_effects[effect] = {"text": str(option)}
                self.logger.debug(f'Text arg = {user_effects[effect]["text"]}')
                continue

            if isinstance(option, list):
                constraint_names = list(constraints.keys())
                user_effects[effect] = {}
                for i, item in enumerate(option):
                    if i < len(constraint_names):
                        user_effects[effect][constraint_names[i]] = item
            else:
                if len(constraints) == 1:
                    user_effects[effect] = {list(constraints.keys())[0]: option}
                else:
                    self.logger.warning(
                        f"Multiple constraints for effect {effect} but only one value provided: {option}"
                    )
                    user_effects[effect] = {list(constraints.keys())[0]: option}

        return user_effects

    def get_image_path(
        self, args: argparse.Namespace, image_input: Optional[Union[str, Path]]
    ) -> str:
        return str(image_input) if image_input else INPUT_IMAGE

    def _get_random_image_path(self):
        images = list(Path(IMAGES_FOLDER).glob("*.*"))
        if not images:
            raise ValueError(f"No images found in {IMAGES_FOLDER}")
        selected_path = str(random.choice(images))
        print("Image selected:", selected_path)
        return selected_path

    def add_special_params(self, result: ParsedCommand, args: argparse.Namespace):
        if "animate" in result.effects:
            result.animation_params = self._create_animation_params(args)
        elif result.command == "ascii":
            result.ascii_params = self._create_ascii_params(args)
        result.output_params = self._create_output_params(args)

    def _create_animation_params(self, args: argparse.Namespace) -> AnimationParams:
        return AnimationParams(
            frames=getattr(args, "frames", 30),
            fps=getattr(args, "fps", 30),
            video_crf=getattr(args, "video_crf", 23),
            video_preset=getattr(args, "video_preset", "medium"),
            gif_duration=getattr(args, "gif_duration", 5),
        )

    def _create_ascii_params(self, args: argparse.Namespace) -> ASCIIParams:
        return ASCIIParams(
            cols=getattr(args, "cols", 100),
            scale=getattr(args, "scale", 0.43),
            font_size=getattr(args, "font_size", 10),
        )

    def _create_output_params(self, args: argparse.Namespace) -> Dict:
        return {}

    def _validate_command(self, result: ParsedCommand):
        pass


'''
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

                print(effect)

                if effect == 'random':
                    parser.add_argument(
                            f"--{effect}",
                            action="store_true",
                            help=help_msg
                    )
                elif effect == 'impact':
                    parser.add_argument(
                            f"--{effect}",
                            nargs="*",
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
        parser.add_argument(
            "--impact",
            nargs='+',
            help="Memerizer..."
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

        print(f"command_str: {command_str}")
        arg_list = shlex.split(command_str)
        print(f'arg_list = {arg_list}')
        command = arg_list[0]
        print(f"Command = {command}")

        try:
            parser = self.parser_hook[command]
        except Exception as e:
            print("parser_hook error")
            raise ValueError(f"Invalid command: {str(e)}")

        try:
            args = parser.parse_args(arg_list[1:])
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
            print(f"my_args: {my_args.items()}")

            for effect, option in my_args.items():
                constraints = self.effect_params[effect]["constraints"]

                # Special handling for impact effect
                if effect == 'impact':
                    # Handle both string and list inputs for impact text
                    if isinstance(option, list):
                        user_effects[effect] = {'text': ' '.join(option)}
                    else:
                        user_effects[effect] = {'text': str(option)}
                    print(f'Text arg = {user_effects[effect]["text"]}')
                    continue

                # Handle list options for other effects
                if isinstance(option, list):
                    constraint_names = list(constraints.keys())
                    user_effects[effect] = {}
                    for i, item in enumerate(option):
                        if i < len(constraint_names):
                            user_effects[effect][constraint_names[i]] = item

                # Handle single value options
                else:
                    if len(constraints) == 1:
                        user_effects[effect] = {list(constraints.keys())[0]: option}
                    else:
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
        images = list(Path(IMAGES_FOLDER).glob("*.*"))
        if not images:
            raise ValueError(f"No images found in {IMAGES_FOLDER}")
        selected_path = str(random.choice(images))
        print("Image selected:", selected_path)
        return selected_path

    def _validate_command(self, parsed: ParsedCommand) -> None:
        if (
            not parsed.image_path
            and not any("--random" in effect[0] for effect in parsed.effects)
        ):
            raise ValueError("No image input specified")
'''
