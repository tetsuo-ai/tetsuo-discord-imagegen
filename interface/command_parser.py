import re
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import logging
from ..config.config import ConfigManager
from ..core.utils import ImageUtils

@dataclass
class ParsedCommand:
    """Structured representation of a parsed command."""
    command: str
    image_path: Optional[str]
    effects: List[Tuple[str, Dict[str, Any]]]
    animation_params: Dict[str, Any]
    ascii_params: Dict[str, Any]
    output_params: Dict[str, Any]
    preset_name: Optional[str]
    tags: List[str]

class CommandParser:
    """
    Parses and validates user commands for image processing.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize command parser with configuration.
        
        Args:
            config: ConfigManager instance for validation
        """
        self.config = config
        self.logger = logging.getLogger('CommandParser')
        
        # Command patterns
        self.patterns = {
            'effect': r'--(\w+)(?:\s+(\d*\.?\d+)|\s+\[([^\]]+)\])',
            'animation': r'--(?:frames|fps)\s+(\d+)',
            'ascii': r'--(?:cols|scale)\s+(\d*\.?\d+)',
            'tag': r'#(\w+)',
            'preset': r'--preset\s+(\w+)',
            'alpha': r'--(?:alpha|coloralpha|rgbalpha)\s+(\d+)',
            'output': r'--(?:format|quality)\s+(\w+)'
        }

    def parse_command(self, command_str: str, 
                     image_input: Optional[Union[str, Path]] = None) -> ParsedCommand:
        """
        Parse command string into structured format.
        
        Args:
            command_str: Raw command string
            image_input: Optional path to input image
            
        Returns:
            ParsedCommand: Structured command data
            
        Raises:
            ValueError: If command is invalid
        """
        # Initialize result structure
        result = ParsedCommand(
            command="process",  # Default command
            image_path=str(image_input) if image_input else None,
            effects=[],
            animation_params={},
            ascii_params={},
            output_params={},
            preset_name=None,
            tags=[]
        )
        
        # Split command into parts
        parts = command_str.split()
        
        # Extract command type if present
        if parts and not parts[0].startswith('--'):
            result.command = parts.pop(0)
        
        # Process remaining parts
        i = 0
        while i < len(parts):
            part = parts[i]
            
            # Handle preset
            if match := re.match(self.patterns['preset'], ' '.join(parts[i:])):
                preset_name = match.group(1)
                try:
                    preset = self.config.get_preset(preset_name)
                    result.preset_name = preset_name
                    result.effects.extend([
                        (effect, params) 
                        for effect, params in preset['params'].items()
                    ])
                    i += 2
                    continue
                except KeyError:
                    raise ValueError(f"Unknown preset: {preset_name}")
            
            # Handle effects
            if match := re.match(self.patterns['effect'], ' '.join(parts[i:])):
                effect_name = match.group(1)
                if effect_name in self.config.effect_params:
                    # Parse effect parameters
                    if match.group(2):  # Single value
                        value = float(match.group(2))
                        params = {'intensity': value} if 'intensity' in self.config.effect_params[effect_name] else {'value': value}
                    elif match.group(3):  # Multiple values
                        values = [float(x.strip()) for x in match.group(3).split(',')]
                        if len(values) == 2:
                            params = {'intensity': tuple(values)} if 'intensity' in self.config.effect_params[effect_name] else {'value': tuple(values)}
                        else:
                            params = {'values': values}
                    
                    # Validate parameters
                    self.config.validate_params(effect_name, params)
                    result.effects.append((effect_name, params))
                    i += 2
                    continue
            
            # Handle animation parameters
            if match := re.match(self.patterns['animation'], ' '.join(parts[i:])):
                param_name = parts[i][2:]  # Remove '--'
                value = int(match.group(1))
                
                if param_name == 'frames':
                    if not (self.config.animation.min_frames <= value <= self.config.animation.max_frames):
                        raise ValueError(f"Frames must be between {self.config.animation.min_frames} and {self.config.animation.max_frames}")
                elif param_name == 'fps':
                    if not (self.config.animation.min_fps <= value <= self.config.animation.max_fps):
                        raise ValueError(f"FPS must be between {self.config.animation.min_fps} and {self.config.animation.max_fps}")
                
                result.animation_params[param_name] = value
                i += 2
                continue
            
            # Handle ASCII parameters
            if match := re.match(self.patterns['ascii'], ' '.join(parts[i:])):
                param_name = parts[i][2:]  # Remove '--'
                value = float(match.group(1))
                
                if param_name == 'cols':
                    if not (0 < value <= self.config.ascii.max_cols):
                        raise ValueError(f"Columns must be between 1 and {self.config.ascii.max_cols}")
                elif param_name == 'scale':
                    if not (0 < value <= 2.0):
                        raise ValueError("Scale must be between 0 and 2.0")
                
                result.ascii_params[param_name] = value
                i += 2
                continue
            
            # Handle alpha values
            if match := re.match(self.patterns['alpha'], ' '.join(parts[i:])):
                param_name = parts[i][2:]  # Remove '--'
                value = int(match.group(1))
                
                if not (0 <= value <= 255):
                    raise ValueError("Alpha value must be between 0 and 255")
                
                result.output_params[param_name] = value
                i += 2
                continue
            
            # Handle output parameters
            if match := re.match(self.patterns['output'], ' '.join(parts[i:])):
                param_name = parts[i][2:]  # Remove '--'
                value = parts[i + 1]
                
                if param_name == 'format':
                    if value.upper() not in ['PNG', 'JPEG', 'GIF']:
                        raise ValueError("Supported formats: PNG, JPEG, GIF")
                elif param_name == 'quality':
                    value = int(value)
                    if not (0 <= value <= 100):
                        raise ValueError("Quality must be between 0 and 100")
                
                result.output_params[param_name] = value
                i += 2
                continue
            
            # Handle tags
            if match := re.match(self.patterns['tag'], part):
                result.tags.append(match.group(1))
                i += 1
                continue
            
            # Handle random flag
            if part == '--random':
                result.image_path = None  # Will be handled by processor
                i += 1
                continue
            
            # Unknown parameter
            raise ValueError(f"Unknown parameter: {part}")
        
        # Validate command
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
        # Validate command type
        if parsed.command not in ['process', 'animate', 'ascii']:
            raise ValueError(f"Unknown command: {parsed.command}")
        
        # Check for required image input
        if not parsed.image_path and not parsed.preset_name and '--random' not in parsed.command:
            raise ValueError("No image input specified")
        
        # Validate animation parameters
        if parsed.command == 'animate':
            if not parsed.animation_params and not parsed.preset_name:
                # Set defaults
                parsed.animation_params['frames'] = self.config.animation.default_frames
                parsed.animation_params['fps'] = self.config.animation.default_fps
        
        # Validate ASCII parameters
        if parsed.command == 'ascii':
            if not parsed.ascii_params:
                # Set defaults
                parsed.ascii_params['cols'] = self.config.ascii.default_cols
                parsed.ascii_params['scale'] = self.config.ascii.default_scale

    def format_help(self) -> str:
        """
        Generate help text for available commands.
        
        Returns:
            str: Formatted help text
        """
        help_text = [
            "Available Commands:",
            "  process [options] - Process image with effects",
            "  animate [options] - Create animation",
            "  ascii [options] - Generate ASCII art",
            "",
            "Effect Options:"
        ]
        
        # Add effect parameters
        for effect, params in self.config.effect_params.items():
            param_desc = []
            for param, config in params.items():
                if isinstance(config, dict) and 'min' in config:
                    param_desc.append(
                        f"{param}=[{config['min']}-{config['max']}]"
                    )
            help_text.append(
                f"  --{effect} {' '.join(param_desc)}"
            )
        
        # Add other options
        help_text.extend([
            "",
            "Animation Options:",
            f"  --frames [{self.config.animation.min_frames}-{self.config.animation.max_frames}]",
            f"  --fps [{self.config.animation.min_fps}-{self.config.animation.max_fps}]",
            "",
            "ASCII Options:",
            f"  --cols [1-{self.config.ascii.max_cols}]",
            "  --scale [0.1-2.0]",
            "",
            "Other Options:",
            "  --preset <name> - Use predefined effect combination",
            "  --format <PNG|JPEG|GIF> - Set output format",
            "  --quality [0-100] - Set output quality",
            "  --random - Use random image from library",
            "  #tag - Add tag to output"
        ])
        
        return "\n".join(help_text)

    def get_example_commands(self) -> List[str]:
        """
        Get list of example commands.
        
        Returns:
            List[str]: Example commands
        """
        return [
            "process --glitch 0.5 --chroma 0.3 #cyberpunk",
            "animate --preset psychic --frames 30 --fps 24",
            "ascii --cols 120 --scale 0.5",
            "process --random --preset cyberpunk",
            "animate --glitch [0.3,0.8] --chroma [0.2,0.4]"
        ]