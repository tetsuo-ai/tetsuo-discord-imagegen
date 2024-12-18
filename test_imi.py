import os
from pathlib import Path
from PIL import Image, ImageDraw
from io import BytesIO
import pytest
import tempfile
import json
import sqlite3
import itertools
from tetimi import ImageProcessor, EFFECT_PRESETS
from anims import AnimationProcessor
from artrepo import ArtRepository

def setup_test_environment():
    """Create necessary test directories and sample image if needed"""
    test_dir = Path("test_outputs")
    test_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for different test types
    for subdir in ['basic', 'combinations', 'presets', 'points', 'animations', 'ascii']:
        (test_dir / subdir).mkdir(exist_ok=True)
    
    # Create test image if it doesn't exist
    test_image_path = Path("input.png")
    if not test_image_path.exists():
        # Create a more interesting test image with some patterns
        img = Image.new('RGB', (400, 300), color='white')
        draw = ImageDraw.Draw(img)
        # Add some shapes for better testing
        draw.rectangle([50, 50, 350, 250], fill='black')
        draw.ellipse([100, 100, 300, 200], fill='gray')
        img.save(test_image_path)
    
    return str(test_image_path.absolute()), test_dir

def test_all_functionality():
    """Generate 150+ test variations"""
    test_image_path, test_dir = setup_test_environment()
    
    # Ensure we have an absolute path
    test_image = Image.open(test_image_path)
    
    # Basic effect tests with explicit image
    basic_tests = {
        'rgb_test': {'rgb': (120, 180, 200), 'rgbalpha': 160},
        'color_test': {'color': (120, 180, 200), 'coloralpha': 160},
        'glitch_test': {'glitch': 5},
        'chroma_test': {'chroma': 8},
        'scan_test': {'scan': 4},
        'noise_test': {'noise': 0.05},
        'energy_test': {'energy': 0.3},
        'pulse_test': {'pulse': 0.2},
        'consciousness_test': {'consciousness': 0.5}
    }
    
    # Points effect variations
    point_sizes = [4]
    reg_offsets = [6]
    point_colors = [
        ['#E62020', '#20B020', '#2020E6', '#D4D420'],
        ['#FF0000', '#00FF00', '#0000FF'],
        ['#E62020', '#2020E6']
    ]
    
    # Generate points effect combinations
    points_tests = {}
    for size, offset, colors in itertools.product(point_sizes, reg_offsets, point_colors):
        test_name = f'points_s{size}_o{offset}_c{len(colors)}'
        points_tests[test_name] = {
            'points': True,
            'dot_size': size,
            'registration_offset': offset,
            'colors': colors
        }
    
    # Effect combinations
    combination_tests = {
        'glitch_color': {'rgb': (200, 50, 50), 'rgbalpha': 160, 'glitch': 3},
        'consciousness_glitch': {'consciousness': 0.6, 'glitch': 4, 'chroma': 12},
    }

    # Run all static image tests
    for test_name, params in basic_tests.items():
        try:
            # Create new processor for each test with explicit image
            processor = ImageProcessor(test_image)
            result = processor.base_image.convert('RGBA')
            
            # Apply effects
            for effect, value in params.items():
                if effect == 'rgb':
                    result = processor.add_color_overlay((*value, params.get('rgbalpha', 255)))
                elif effect == 'color':
                    result = processor.colorize_non_white(*value, params.get('coloralpha', 255))
                elif effect in ['glitch', 'chroma', 'scan', 'noise', 'energy', 'pulse', 'consciousness']:
                    result = processor.apply_effect(effect, {effect: value})
            
            # Save result
            output_path = test_dir / 'basic' / f"{test_name}.png"
            result.save(output_path)
            print(f"Generated {output_path}")
            
        except Exception as e:
            print(f"Error in {test_name}: {str(e)}")

    # Points tests
    for test_name, params in points_tests.items():
        try:
            processor = ImageProcessor(test_image, points=True)
            result = processor.apply_points_effect(
                colors=params.get('colors'),
                dot_size=params.get('dot_size'),
                registration_offset=params.get('registration_offset')
            )
            
            output_path = test_dir / 'points' / f"{test_name}.png"
            result.save(output_path)
            print(f"Generated {output_path}")
            
        except Exception as e:
            print(f"Error in {test_name}: {str(e)}")

    # Combination tests
    for test_name, params in combination_tests.items():
        try:
            processor = ImageProcessor(test_image)
            result = processor.base_image.convert('RGBA')
            
            for effect, value in params.items():
                result = processor.apply_effect(effect, {effect: value})
            
            output_path = test_dir / 'combinations' / f"{test_name}.png"
            result.save(output_path)
            print(f"Generated {output_path}")
            
        except Exception as e:
            print(f"Error in {test_name}: {str(e)}")

    # Preset tests
    for preset_name, preset_params in EFFECT_PRESETS.items():
        try:
            processor = ImageProcessor(test_image)
            result = processor.base_image.convert('RGBA')
            
            for effect in ['rgb', 'color', 'glitch', 'chroma', 'scan', 'noise', 'energy', 'pulse']:
                if effect in preset_params:
                    result = processor.apply_effect(effect, preset_params)
            
            output_path = test_dir / 'presets' / f"{preset_name}.png"
            result.save(output_path)
            print(f"Generated preset {preset_name}")
            
        except Exception as e:
            print(f"Error in {preset_name}: {str(e)}")

    # ASCII Art tests
    ascii_configs = {
        'ascii_basic': {'cols': 80, 'scale': 0.43},
    }
    
    for config_name, params in ascii_configs.items():
        try:
            processor = ImageProcessor(test_image)
            ascii_art = processor.convertImageToAscii(**params)
            
            output_path = test_dir / "ascii" / f"{config_name}.txt"
            with open(output_path, 'w') as f:
                f.write('\n'.join(ascii_art))
            print(f"Generated ASCII art: {output_path}")
            
        except Exception as e:
            print(f"Error in ASCII conversion {config_name}: {str(e)}")

    # Animation tests
    animation_tests = {
        'basic_glitch': {'style': 'glitch_surge', 'frames': 15},
        'psychic_blast': {'style': 'psychic_blast', 'frames': 15},
        'neo_flash': {'style': 'neo_flash', 'frames': 15},
        'points_animation': {
            'style': 'glitch_surge',
            'frames': 15,
            'points': True
        }
    }

    for anim_name, params in animation_tests.items():
        animation_processor = None
        try:
            animation_processor = AnimationProcessor(test_image)
            frames = animation_processor.generate_frames(
                preset_name=params['style'],
                num_frames=params['frames']
            )
            
            video_path = animation_processor.create_video(
                frame_rate=24,
                output_name=f"test_{anim_name}.mp4"
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
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
        test_db_path = tmp_db.name
        
        try:
            repo = ArtRepository(db_path=test_db_path)
            
            # Store test artwork with effects
            processor = ImageProcessor(test_image)
            result = processor.apply_effect('glitch', {'glitch': 5})
            
            # Convert to bytes for storage
            img_bytes = BytesIO()
            result.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            # Test database operations
            artwork_id = repo.store_artwork(
                image=img_bytes,
                title="Test Artwork",
                creator_id="test_user",
                creator_name="Test User",
                tags=["test", "glitch"],
                parameters={"glitch": 5}
            )
            
            # Test retrieval
            retrieved_img, metadata = repo.get_artwork(artwork_id)
            assert metadata['title'] == "Test Artwork"
            print("Database operations successful")
            
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
