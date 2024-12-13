import os
from pathlib import Path
from PIL import Image
from io import BytesIO
import pytest
from tetimi import ImageProcessor, EFFECT_PRESETS

def setup_test_environment():
    """Create necessary test directories and sample image if needed"""
    # Create test directory
    test_dir = Path("test_outputs")
    test_dir.mkdir(exist_ok=True)
    
    # Create a simple test image if it doesn't exist
    test_image_path = Path("input.png")
    if not test_image_path.exists():
        img = Image.new('RGB', (400, 300), color='white')
        img.save(test_image_path)
    
    return test_image_path, test_dir

def test_basic_effects():
    """Test each individual effect with moderate parameters"""
    test_image_path, test_dir = setup_test_environment()
    
    # Test cases with moderate parameters to avoid overwhelming effects
    test_cases = {
        'rgb_test': {'rgb': (120, 180, 200), 'alpha': 160},
        'glitch_test': {'glitch': 5},  # Reduced from default
        'chroma_test': {'chroma': 8},  # More subtle chromatic aberration
        'scan_test': {'scan': 4},  # Wider scan lines
        'noise_test': {'noise': 0.05},  # Very subtle noise
        'energy_test': {'energy': 0.3},  # Reduced energy effect
        'pulse_test': {'pulse': 0.2}  # Subtle pulse
    }
    
    for test_name, params in test_cases.items():
        try:
            processor = ImageProcessor(test_image_path)
            
            # Apply the effect
            result = processor.base_image.convert('RGBA')
            
            if 'rgb' in params:
                r, g, b = params['rgb']
                alpha = params.get('alpha', 255)
                result = processor.colorize_non_white(r, g, b, alpha)
            elif 'glitch' in params:
                result = processor.apply_glitch_effect(params['glitch'])
            elif 'chroma' in params:
                result = processor.add_chromatic_aberration(params['chroma'])
            elif 'scan' in params:
                result = processor.add_scan_lines(params['scan'])
            elif 'noise' in params:
                result = processor.add_noise(params['noise'])
            elif 'energy' in params:
                result = processor.apply_energy_effect(params['energy'])
            elif 'pulse' in params:
                result = processor.apply_pulse_effect(params['pulse'])
            
            # Save the result
            output_path = test_dir / f"{test_name}.png"
            result.save(output_path)
            print(f"Generated {output_path}")
            
        except Exception as e:
            print(f"Error in {test_name}: {str(e)}")

def test_presets():
    """Test each preset with default parameters"""
    test_image_path, test_dir = setup_test_environment()
    
    for preset_name, preset_params in EFFECT_PRESETS.items():
        try:
            processor = ImageProcessor(test_image_path)
            result = processor.base_image.convert('RGBA')
            
            # Apply preset effects in order
            if 'rgb' in preset_params:
                r, g, b = preset_params['rgb']
                alpha = preset_params.get('alpha', 255)
                result = processor.colorize_non_white(r, g, b, alpha)
            
            for effect in ['glitch', 'chroma', 'scan', 'noise', 'energy', 'pulse']:
                if effect in preset_params:
                    processor.base_image = result.copy()
                    if effect == 'glitch':
                        result = processor.apply_glitch_effect(preset_params[effect])
                    elif effect == 'chroma':
                        result = processor.add_chromatic_aberration(preset_params[effect])
                    elif effect == 'scan':
                        result = processor.add_scan_lines(preset_params[effect])
                    elif effect == 'noise':
                        result = processor.add_noise(preset_params[effect])
                    elif effect == 'energy':
                        result = processor.apply_energy_effect(preset_params[effect])
                    elif effect == 'pulse':
                        result = processor.apply_pulse_effect(preset_params[effect])
            
            # Save the result
            output_path = test_dir / f"preset_{preset_name}.png"
            result.save(output_path)
            print(f"Generated preset {preset_name}")
            
        except Exception as e:
            print(f"Error in preset {preset_name}: {str(e)}")

def test_parameter_ranges():
    """Test effects with different parameter values to find sweet spots"""
    test_image_path, test_dir = setup_test_environment()
    
    # Test ranges for each effect
    test_ranges = {
        'rgb': [(120, 180, 200), (200, 100, 150), (50, 150, 200)],
        'alpha': [120, 160, 200],
        'glitch': [3, 5, 8],
        'chroma': [5, 8, 12],
        'scan': [3, 5, 8],
        'noise': [0.02, 0.05, 0.08],
        'energy': [0.2, 0.3, 0.4],
        'pulse': [0.15, 0.25, 0.35]
    }
    
    for effect, values in test_ranges.items():
        for i, value in enumerate(values):
            try:
                processor = ImageProcessor(test_image_path)
                result = processor.base_image.convert('RGBA')
                
                if effect == 'rgb':
                    result = processor.colorize_non_white(*value, 160)
                elif effect == 'glitch':
                    result = processor.apply_glitch_effect(value)
                elif effect == 'chroma':
                    result = processor.add_chromatic_aberration(value)
                elif effect == 'scan':
                    result = processor.add_scan_lines(value)
                elif effect == 'noise':
                    result = processor.add_noise(value)
                elif effect == 'energy':
                    result = processor.apply_energy_effect(value)
                elif effect == 'pulse':
                    result = processor.apply_pulse_effect(value)
                
                output_path = test_dir / f"{effect}_range_{i}.png"
                result.save(output_path)
                print(f"Generated {effect} test {i}")
                
            except Exception as e:
                print(f"Error in {effect} test {i}: {str(e)}")

if __name__ == "__main__":
    print("Starting ImageProcessor tests...")
    print("\nTesting basic effects...")
    test_basic_effects()
    
    print("\nTesting presets...")
    test_presets()
    
    print("\nTesting parameter ranges...")
    test_parameter_ranges()
    
    print("\nTests completed. Check the test_outputs directory for results.")