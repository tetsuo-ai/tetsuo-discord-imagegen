import itertools
import json
import os
import sqlite3
import sys
import tempfile
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from anims import AnimationProcessor
from artrepo import ArtRepository
from tetimi import ANIMATION_PRESETS, ImageProcessor, colorize_non_white


def setup_test_environment():
    """Create necessary test directories and sample image if needed"""
    test_dir = Path("test_outputs")
    test_dir.mkdir(exist_ok=True)

    # Create subdirectories for different test types
    for subdir in [
        "basic",
        "combinations",
        "presets",
        "points",
        "animations",
        "ascii",
        "consciousness",
    ]:
        (test_dir / subdir).mkdir(exist_ok=True)

    # Create test image if it doesn't exist
    test_image_path = Path("input.png")
    if not test_image_path.exists():
        # Create a more interesting test image with some patterns
        img = Image.new("RGB", (400, 300), color="white")
        draw = ImageDraw.Draw(img)
        # Add some shapes for better testing
        draw.rectangle([50, 50, 350, 250], fill="black")
        draw.ellipse([100, 100, 300, 200], fill="gray")
        draw.line([50, 50, 350, 250], fill="white", width=5)
        draw.line([50, 250, 350, 50], fill="white", width=5)
        img.save(test_image_path)

    return str(test_image_path.absolute()), test_dir


def test_all_functionality():
    """Generate comprehensive test variations for all effects and commands"""
    test_image_path, test_dir = setup_test_environment()
    test_image = Image.open(test_image_path)

    # Basic effect tests
    basic_tests = {
        "rgb_test": {"rgb": (120, 180, 200), "rgbalpha": 160},
        "color_test": {"color": (120, 180, 200), "coloralpha": 160},
        "glitch_test": {"glitch": 5},
        "chroma_test": {"chroma": 8},
        "scan_test": {"scan": 4},
        "noise_test": {"noise": 0.05},
        "energy_test": {"energy": 0.3},
        "pulse_test": {"pulse": 0.2},
        "consciousness_test": {"consciousness": 0.5},
        "impact_test": {"impact": "TEST IMAGE"},
    }

    # Points effect variations
    point_colors = [
        ["#E62020", "#20B020", "#2020E6", "#D4D420"],  # 4 colors
        ["#FF0000", "#00FF00", "#0000FF"],  # RGB
        ["#E62020", "#2020E6"],  # 2 colors
    ]

    points_configs = [
        {"dot_size": 4, "registration_offset": 6},
        {"dot_size": 6, "registration_offset": 8},
        {"dot_size": 3, "registration_offset": 4},
    ]

    # Generate points effect combinations
    points_tests = {}
    for config, colors in itertools.product(points_configs, point_colors):
        test_name = f'points_s{config["dot_size"]}_o{config["registration_offset"]}_c{len(colors)}'
        points_tests[test_name] = {
            "points": True,
            "dot_size": config["dot_size"],
            "registration_offset": config["registration_offset"],
            "colors": colors,
        }

    # Effect combinations with consciousness
    combination_tests = {
        "glitch_color": {"rgb": (200, 50, 50), "rgbalpha": 160, "glitch": 3},
        "consciousness_glitch": {"consciousness": 0.6, "glitch": 4, "chroma": 12},
        "full_effect": {
            "rgb": (180, 50, 200),
            "rgbalpha": 140,
            "glitch": 6,
            "chroma": 15,
            "scan": 100,
            "noise": 0.1,
            "energy": 0.4,
            "pulse": 0.3,
            "consciousness": 0.7,
            "impact": "$TETSUO",
        },
    }

    # Run all static image tests
    for test_name, params in basic_tests.items():
        try:
            processor = ImageProcessor(test_image)
            result = processor.base_image.convert("RGBA")

            for effect, value in params.items():
                if effect == "rgb":
                    result = processor.add_color_overlay(
                        (*value, params.get("rgbalpha", 255))
                    )
                elif effect == "color":
                    result = colorize_non_white(
                        processor * value, params.get("coloralpha", 255)
                    )
                elif effect in [
                    "glitch",
                    "chroma",
                    "scan",
                    "noise",
                    "energy",
                    "pulse",
                    "consciousness",
                    "impact",
                ]:
                    result = processor.apply_effect(effect, {effect: value})

            output_path = test_dir / "basic" / f"{test_name}.png"
            result.save(output_path)
            print(f"Generated {output_path}")

        except Exception as e:
            print(f"Error in {test_name}: {str(e)}")

    # Points tests
    """ Commenting out points tests for now
    for test_name, params in points_tests.items():
        try:
            processor = ImageProcessor(test_image, points=True)
            result = processor.apply_points_effect(
                colors=params.get("colors"),
                dot_size=params.get("dot_size"),
                registration_offset=params.get("registration_offset"),
            )

            output_path = test_dir / "points" / f"{test_name}.png"
            try:
                result.save(output_path)
            except Exception as e:
                print("Error saving points image: ", e)
                sys.exit(1)
            print(f"Generated {output_path}")

        except Exception as e:
            print(f"Error in {test_name}: {str(e)}")

    """
    # Combination tests
    for test_name, params in combination_tests.items():
        try:
            processor = ImageProcessor(test_image)
            result = processor.base_image.convert("RGBA")

            for effect in [
                "rgb",
                "color",
                "glitch",
                "chroma",
                "scan",
                "noise",
                "energy",
                "pulse",
                "consciousness",
                "impact",
            ]:
                if effect in params:
                    result = processor.apply_effect(effect, {effect: params[effect]})

            output_path = test_dir / "combinations" / f"{test_name}.png"
            result.save(output_path)
            print(f"Generated {output_path}")

        except Exception as e:
            print(f"Error in {test_name}: {str(e)}")

    # Preset tests
    for preset_name, preset_params in ANIMATION_PRESETS.items():
        try:
            processor = ImageProcessor(test_image)
            result = processor.base_image.convert("RGBA")

            for effect in [
                "rgb",
                "color",
                "glitch",
                "chroma",
                "scan",
                "noise",
                "energy",
                "pulse",
                "consciousness",
                "impact",
            ]:
                if effect in preset_params:
                    result = processor.apply_effect(effect, preset_params)

            output_path = test_dir / "presets" / f"{preset_name}.png"
            result.save(output_path)
            print(f"Generated preset {preset_name}")

        except Exception as e:
            print(f"Error in {preset_name}: {str(e)}")

    # ASCII Art tests with different configurations
    ascii_configs = {
        "ascii_basic": {"cols": 80, "scale": 0.43},
        "ascii_detailed": {"cols": 120, "scale": 0.43},
        "ascii_compact": {"cols": 60, "scale": 0.43},
    }

    for config_name, params in ascii_configs.items():
        try:
            processor = ImageProcessor(test_image)
            ascii_art = processor.convertImageToAscii(**params)

            output_path = test_dir / "ascii" / f"{config_name}.txt"
            with open(output_path, "w") as f:
                f.write("\n".join(ascii_art))
            print(f"Generated ASCII art: {output_path}")

        except Exception as e:
            print(f"Error in ASCII conversion {config_name}: {str(e)}")

    # Animation tests with presets and effects
    animation_tests = {
        "basic_glitch": {"glitch": 10, "frames": 15},
        "consciousness_flow": {"consciousness": 0.8, "frames": 15},
        "energy_pulse": {"energy": 0.6, "pulse": 0.4, "frames": 15},
        "full_effect": {
            "glitch": 8,
            "chroma": 15,
            "scan": 100,
            "noise": 0.1,
            "consciousness": 0.6,
            "frames": 15,
        },
    }

    # Add preset-based animation tests EFFECT_PRESETS not defined, commenting out.
    for preset_name in ANIMATION_PRESETS:
        animation_tests[f"preset_{preset_name}"] = {
            **ANIMATION_PRESETS[preset_name],
            "frames": 15,
        }

    for anim_name, params in animation_tests.items():
        animation_processor = None
        try:
            animation_processor = AnimationProcessor(test_image)
            frames = animation_processor.generate_frames(
                params, num_frames=params.pop("frames", 15)
            )

            video_path = animation_processor.create_video(
                frame_rate=24, output_name=f"test_{anim_name}.mp4"
            )
            print(f"Generated animation {anim_name}")

        except Exception as e:
            print(f"Error in animation {anim_name}: {str(e)}")
        finally:
            if animation_processor:
                animation_processor.cleanup()


def test_database():
    """Test database operations"""
    test_image_path, _ = setup_test_environment()
    test_image = Image.open(test_image_path)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        test_db_path = tmp_db.name

        try:
            repo = ArtRepository(db_path=test_db_path)

            # Test each effect type
            effects_to_test = {
                "glitch": {"glitch": 5},
                "consciousness": {"consciousness": 0.6},
                "points": {"points": True, "dot_size": 4, "registration_offset": 6},
                # "preset": ANIMATION_PRESETS["cyberpunk"],
            }

            for effect_name, params in effects_to_test.items():
                # Process image with effect
                processor = ImageProcessor(test_image)

                if effect_name == "points":
                    """Commenting out points_effects
                    result = processor.apply_points_effect(
                        dot_size=params["dot_size"],
                        registration_offset=params["registration_offset"],
                    )
                    """
                    continue
                else:
                    result = processor.apply_effect(effect_name, params)

                # Convert to bytes for storage
                img_bytes = BytesIO()
                result.save(img_bytes, format="PNG")
                img_bytes.seek(0)

                # Store in database
                artwork_id = repo.store_artwork(
                    image=img_bytes,
                    title=f"Test {effect_name.title()}",
                    creator_id="test_user",
                    creator_name="Test User",
                    tags=[effect_name, "test"],
                    parameters=params,
                )

                # Test retrieval
                retrieved_img, metadata = repo.get_artwork(artwork_id)
                assert metadata["title"] == f"Test {effect_name.title()}"
                print(f"Database operation successful for {effect_name}")

        except Exception as e:
            print(f"Database test error: {str(e)}")
        finally:
            if os.path.exists(test_db_path):
                os.unlink(test_db_path)


if __name__ == "__main__":
    print("Starting comprehensive ImageProcessor tests...")
    test_all_functionality()
    test_database()
    print("Tests completed.")
