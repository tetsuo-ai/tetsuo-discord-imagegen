import os
from pathlib import Path
from PIL import Image
from io import BytesIO
import pytest
from tetimi import ImageProcessor, EFFECT_PRESETS

def setup_test_environment():
    """Create necessary test directories and sample image if needed"""
    test_dir = Path("test_outputs")
    test_dir.mkdir(exist_ok=True)
    
    test_image_path = Path("input.png")
    if not test_image_path.exists():
        img = Image.new('RGB', (400, 300), color='white')
        img.save(test_image_path)
    
    return test_image_path, test_dir

def test_basic_effects():
    """Test each individual effect with moderate parameters"""
    test_image_path, test_dir = setup_test_environment()
    
    test_cases = {
        'rgb_test': {'rgb': (120, 180, 200), 'rgbalpha': 160},
        'color_test': {'color': (120, 180, 200), 'coloralpha': 160},
        'glitch_test': {'glitch': 5},
        'chroma_test': {'chroma': 8},
        'scan_test': {'scan': 4},
        'noise_test': {'noise': 0.05},
        'energy_test': {'energy': 0.3},
        'pulse_test': {'pulse': 0.2}
    }
    
    for test_name, params in test_cases.items():
        try:
            processor = ImageProcessor(test_image_path)
            result = processor.base_image.convert('RGBA')
            
            if 'rgb' in params:
                r, g, b = params['rgb']
                alpha = params.get('rgbalpha', 255)
                result = processor.add_color_overlay((r, g, b, alpha))
            elif 'color' in params:
                r, g, b = params['color']
                alpha = params.get('coloralpha', 255)
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
            
            for effect in ['rgb', 'color', 'glitch', 'chroma', 'scan', 'noise', 'energy', 'pulse']:
                if effect in preset_params:
                    processor.base_image = result.copy()
                    if effect == 'rgb':
                        r, g, b = preset_params['rgb']
                        alpha = preset_params.get('rgbalpha', 255)
                        result = processor.add_color_overlay((r, g, b, alpha))
                    elif effect == 'color':
                        r, g, b = preset_params['color']
                        alpha = preset_params.get('coloralpha', 255)
                        result = processor.colorize_non_white(r, g, b, alpha)
                    elif effect == 'glitch':
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
            
            output_path = test_dir / f"preset_{preset_name}.png"
            result.save(output_path)
            print(f"Generated preset {preset_name}")
            
        except Exception as e:
            print(f"Error in preset {preset_name}: {str(e)}")

if __name__ == "__main__":
    print("Starting ImageProcessor tests...")
    test_basic_effects()
    test_presets()
    print("Tests completed. Check the test_outputs directory for results.")