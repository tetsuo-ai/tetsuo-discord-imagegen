import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, Union

import yaml
from dotenv import load_dotenv


@dataclass
class AnimationConfig:
    """Configuration settings for animations."""

    min_frames: int = 15
    max_frames: int = 120
    default_frames: int = 30
    min_fps: int = 10
    max_fps: int = 60
    default_fps: int = 24
    video_crf: int = 23
    video_preset: str = "medium"
    gif_duration: int = 50


@dataclass
class ASCIIConfig:
    """Configuration settings for ASCII art generation."""

    default_cols: int = 80
    max_cols: int = 200
    default_scale: float = 0.43
    default_font_size: int = 14
    basic_chars: str = " .:-=+*#%@"
    detailed_chars: str = " .,:;irsXA253hMHGS#9B&@"


@dataclass
class ProcessingConfig:
    """Configuration settings for image processing."""

    max_image_size: int = 4096
    jpeg_quality: int = 85
    png_compression: int = 6
    default_format: str = "PNG"


class ConfigManager:
    """
    Manages configuration settings and effect presets for the image processing system.
    """

    def __init__(self, config_dir: Union[str, Path] = "config"):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory containing configuration files
        """
        self.logger = logging.getLogger("ConfigManager")
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        load_dotenv()
        self.DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
        self.IMAGES_FOLDER = "images"
        self.INPUT_IMAGE = "input.png"

        # Initialize configuration objects
        self.effect_params = self._build_effect_argument()

        self.animation = AnimationConfig()
        self.ascii = ASCIIConfig()
        self.processing = ProcessingConfig()

        # Load effect configurations
        self.effect_order = [
            "rgb",
            "color",
            "glitch",
            "chroma",
            "scan",
            "noise",
            "energy",
            "pulse",
            "consciousness",
        ]

        # Effect paramter value ranges.

    def _build_effect_argument(self) -> Dict[str, Dict[str, Any]]:
        effect_params = Dict[str, Dict[str, Any]]

        effect_params = {
            "random": {
                "constraints": {
                    None: None
                },
                "description": "Random Image Selection",
            },
            "glitch": {
                "constraints": {
                    "intensity": {"type": float, "min": 0.0, "max": 1.0,
                                  "default": 0.5},
                },
                "description": "Glitch effect intensity",
            },
            "chroma": {
                "constraints": {
                    "intensity": {"type": float, "min": 0.0, "max": 1.0,
                                  "default": 0.5},
                },
                "description": "Chromatic aberration offset",
            },
            "scan": {
                "constraints": {
                    "gap": {"type": int, "min": 1, "max": 50, "default": 2},
                    "opacity": {"type": float, "min": 0.0, "max": 1.0,
                                "default": 0.5},
                },
                "description": "Scan line effect",
            },
            "noise": {
                "constraints": {
                    "intensity": {"type": float, "min": 0.0, "max": 1.0,
                                  "default": 0.5},
                },
                "description": "Noise effect intensity",
            },
            "energy": {
                "constraints": {
                    "intensity": {"type": float, "min": 0.0, "max": 1.0,
                                  "default": 0.5},
                },
                "description": "Energy effect intensity",
            },
            "pulse": {
                "constraints": {
                    "intensity": {"type": float, "min": 0.0, "max": 1.0,
                                  "default": 0.5},
                },
                "description": "Pulse effect intensity",
            },
            "consciousness": {
                "constraints": {
                        "intensity": {"type": float, "min": 0.0, "max": 1.0,
                                      "default": 0.5},
                },
                "description": "Consciousness effect intensity",
            },
            "g": {
                "constraints": {
                    "intensity": {"type": float, "min": 0.3, "max": 1.0,
                                  "default": 0.5},
                },
                "description": "Green movement vector for rgb_gif animation",
            },
            "b": {
                "constraints": {
                    "intensity": {"type": float, "min": 0.3, "max": 1.0,
                                  "default": 0.5},
                },
                "description": "Blue movement vector for rgb_gif animation",
            },
            "impact": {
                "constraints": {
                    "intensity": {"type": str, "min": 1, "max": 50,
                                  "default": "$TETSUO"},
                    "font": {"type": int, "min": 1, "max": 100, "default": 70},
                },
                "description": "Impact effect length",
            },
            "frames": {
                "constraints": {
                    "count": {"type": int, "min": 1, "max": 48, "default": 24},
                },
                "description": "Frame count for animations",
            },
            "fps": {
                "constraints": {
                    "count": {"type": int, "min": 2, "max": 60, "default": 6},
                },
                "description": "Fps count for animation",
            },
            "rgb": {
                "constraints": {
                    "count": {"type": float, "min": 0, "max": 255,
                              "default": 75},
                },
                "description": "RGB color",
            },
            "rgbalpha": {
                "constraints": {
                    # Assuming a typical RGBA structure with an alpha
                    # channel
                    "insensity": {"type": int, "min": 0, "max": 255, "default": 255},
                },
                "description": "RGBA color with alpha channel",
            },
        }

        return effect_params
        # Load presets
        self.presets = self._load_presets()

    def _load_presets(self) -> Dict[str, Dict[str, Any]]:
        """
        Load effect presets from configuration file.

        Returns:
            Dict[str, Dict[str, Any]]: Preset configurations
        """
        preset_path = self.config_dir / "presets.yml"

        if not preset_path.exists():
            # Create default presets if file doesn't exist
            default_presets = {
                "cyberpunk": {
                    "params": {
                        "glitch": {"intensity": (0.3, 0.8)},
                        "chroma": {"offset": (0.2, 0.4)},
                        "scan": {"gap": (2, 4), "opacity": (0.4, 0.6)},
                        "consciousness": {"intensity": (0.3, 0.7)},
                    },
                    "frames": 30,
                    "fps": 24,
                    "description": "Cyberpunk-style glitch effect with scanning",
                },
                "psychic": {
                    "params": {
                        "energy": {"intensity": (0.4, 0.8)},
                        "consciousness": {"intensity": (0.5, 0.9)},
                        "pulse": {"intensity": (0.2, 0.6)},
                    },
                    "frames": 30,
                    "fps": 24,
                    "description": "Psychic energy visualization",
                },
            }

            # Save default presets
            with open(preset_path, "w") as f:
                yaml.dump(default_presets, f)

            return default_presets

        # Load existing presets
        try:
            with open(preset_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading presets: {e}")
            return {}

    def save_preset(
        self, name: str, params: Dict[str, Any], description: Optional[str] = None
    ) -> None:
        """
        Save a new effect preset.

        Args:
            name: Preset name
            params: Effect parameters
            description: Optional preset description
        """
        # Validate parameters against effect constraints
        for effect, effect_params in params.items():
            if effect not in self.effect_params:
                raise ValueError(f"Unknown effect: {effect}")

            constraints = self.effect_params[effect]
            for param, value in effect_params.items():
                if param not in constraints:
                    raise ValueError(
                        f"Unknown parameter '{param}' for effect '{effect}'"
                    )

                param_range = constraints[param]
                if isinstance(value, (tuple, list)):
                    for v in value:
                        if not (param_range["min"] <= v <= param_range["max"]):
                            raise ValueError(
                                f"Value {v} for {effect}.{param} outside valid range "
                                f"[{param_range['min']}, {param_range['max']}]"
                            )
                else:
                    if not (param_range["min"] <= value <= param_range["max"]):
                        raise ValueError(
                            f"Value {value} for {effect}.{param} outside valid range "
                            f"[{param_range['min']}, {param_range['max']}]"
                        )

        # Add preset
        self.presets[name] = {
            "params": params,
            "description": description or f"Custom preset: {name}",
        }

        # Save to file
        preset_path = self.config_dir / "presets.yml"
        with open(preset_path, "w") as f:
            yaml.dump(self.presets, f)

    def get_preset(self, name: str) -> Dict[str, Any]:
        """
        Get preset configuration by name.

        Args:
            name: Preset name

        Returns:
            Dict[str, Any]: Preset configuration

        Raises:
            KeyError: If preset doesn't exist
        """
        if name not in self.presets:
            raise KeyError(f"Preset '{name}' not found")
        return self.presets[name]

    def validate_params(self, effect: str, params: Dict[str, Any]) -> None:
        """
        Validate effect parameters against constraints.

        Args:
            effect: Effect name
            params: Effect parameters

        Raises:
            ValueError: If parameters are invalid
        """
        if effect not in self.effect_params:
            raise ValueError(f"Unknown effect: {effect}")

        constraints = self.effect_params[effect]
        for param, value in params.items():
            if param not in constraints:
                raise ValueError(f"Unknown parameter '{param}' for effect '{effect}'")

            param_range = constraints[param]
            if isinstance(value, (tuple, list)):
                for v in value:
                    if not (param_range["min"] <= v <= param_range["max"]):
                        raise ValueError(
                            f"Value {v} for {effect}.{param} outside valid range "
                            f"[{param_range['min']}, {param_range['max']}]"
                        )
            else:
                if not (param_range["min"] <= value <= param_range["max"]):
                    raise ValueError(
                        f"Value {value} for {effect}.{param} outside valid range "
                        f"[{param_range['min']}, {param_range['max']}]"
                    )

    def get_default_params(self, effect: str) -> Dict[str, Any]:
        """
        Get default parameters for an effect.

        Args:
            effect: Effect name

        Returns:
            Dict[str, Any]: Default parameters

        Raises:
            ValueError: If effect doesn't exist
        """
        if effect not in self.effect_params:
            raise ValueError(f"Unknown effect: {effect}")

        return {
            param: config["default"]
            for param, config in self.effect_params[effect].items()
            if isinstance(config, dict) and "default" in config
        }

    def save_config(self) -> None:
        """Save current configuration to files."""
        config_path = self.config_dir / "config.yml"

        config = {
            "animation": self.animation.__dict__,
            "ascii": self.ascii.__dict__,
            "processing": self.processing.__dict__,
            "effect_order": self.effect_order,
            "effect_params": self.effect_params,
        }

        with open(config_path, "w") as f:
            yaml.dump(config, f)

    def load_config(self) -> None:
        """Load configuration from files."""
        config_path = self.config_dir / "config.yml"

        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)

                # Update configuration objects
                for key, value in config.get("animation", {}).items():
                    setattr(self.animation, key, value)
                for key, value in config.get("ascii", {}).items():
                    setattr(self.ascii, key, value)
                for key, value in config.get("processing", {}).items():
                    setattr(self.processing, key, value)

                self.effect_order = config.get("effect_order", self.effect_order)
                self.effect_params = config.get("effect_params", self.effect_params)

            except Exception as e:
                self.logger.error(f"Error loading configuration: {e}")
