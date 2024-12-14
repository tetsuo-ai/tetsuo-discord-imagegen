from pathlib import Path
import numpy as np
from tetimi import ImageProcessor, EFFECT_PRESETS
import itertools
import time

class PresetOptimizer:
    def __init__(self, test_image_path="input.png", output_dir="preset_analysis"):
        self.test_image_path = Path(test_image_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        
        # Parameter variation ranges for optimization
        self.variation_ranges = {
            'alpha': (-20, 20),  # Vary alpha by ±20
            'glitch': (-2, 2),   # Vary glitch intensity by ±2
            'chroma': (-2, 2),   # Vary chromatic aberration by ±2
            'scan': (-20, 20),   # Vary scan line spacing by ±20
            'noise': (-0.01, 0.01),  # Vary noise by ±0.01
            'energy': (-0.05, 0.05),  # Vary energy by ±0.05
            'pulse': (-0.05, 0.05)    # Vary pulse by ±0.05
        }
        
        # Store results
        self.preset_scores = {}
        self.optimized_presets = {}
        
    def evaluate_preset(self, preset_name, params):
        """Evaluate a single preset configuration"""
        try:
            processor = ImageProcessor(self.test_image_path)
            result = processor.base_image.convert('RGBA')
            
            if 'rgb' in params:
                r, g, b = params['rgb']
                alpha = params.get('alpha', 255)
                result = processor.colorize_non_white(r, g, b, alpha)
            
            for effect in ['glitch', 'chroma', 'noise', 'scan', 'energy', 'pulse']:
                if effect in params:
                    processor.base_image = result.copy()
                    if effect == 'glitch':
                        result = processor.apply_glitch_effect(int(params[effect]))
                    elif effect == 'chroma':
                        result = processor.add_chromatic_aberration(int(params[effect]))
                    elif effect == 'scan':
                        result = processor.add_scan_lines(int(params[effect]))
                    elif effect == 'noise':
                        result = processor.add_noise(float(params[effect]))
                    elif effect == 'energy':
                        result = processor.apply_energy_effect(float(params[effect]))
                    elif effect == 'pulse':
                        result = processor.apply_pulse_effect(float(params[effect]))
            
            # Convert to RGB for analysis
            rgb_result = result.convert('RGB')
            
            # Calculate score using our existing metrics
            stats = ImageStat.Stat(rgb_result)
            means = stats.mean
            stddevs = stats.stddev
            
            # Check if image is too washed out
            is_washed = any(m > 240 for m in means) or all(s < 20 for s in stddevs)
            if is_washed:
                return 0.2, result
            
            # Calculate color variation
            channel_separation = np.std(means) / 255.0
            
            # Edge detection for detail
            edges = rgb_result.filter(ImageFilter.FIND_EDGES)
            edge_score = ImageStat.Stat(edges).mean[0] / 255.0
            
            # Final score calculation
            score = (channel_separation * 0.4 + edge_score * 0.6)
            
            # Save high-scoring variations
            if score > 0.5:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                image_path = self.images_dir / f"{preset_name}_{timestamp}_{score:.3f}.png"
                result.save(image_path)
            
            return score, result
            
        except Exception as e:
            print(f"Error evaluating {preset_name}: {str(e)}")
            return 0.0, None

    def optimize_preset(self, preset_name, preset_params, num_iterations=50):
        """Optimize a preset by trying variations of its parameters"""
        best_score = 0.0
        best_params = preset_params.copy()
        
        print(f"\nOptimizing {preset_name}...")
        print(f"Original parameters: {preset_params}")
        
        original_score, _ = self.evaluate_preset(preset_name, preset_params)
        print(f"Original score: {original_score:.3f}")
        
        for _ in range(num_iterations):
            # Create a variation of the parameters
            variation = preset_params.copy()
            
            # Randomly select 1-3 parameters to vary
            num_params = np.random.randint(1, 4)
            params_to_vary = np.random.choice(list(self.variation_ranges.keys()), 
                                            size=min(num_params, len(self.variation_ranges)),
                                            replace=False)
            
            for param in params_to_vary:
                if param in variation:
                    min_val, max_val = self.variation_ranges[param]
                    delta = np.random.uniform(min_val, max_val)
                    
                    # Apply the variation while respecting parameter bounds
                    if param in ['alpha']:
                        variation[param] = np.clip(variation[param] + delta, 120, 220)
                    elif param in ['glitch']:
                        variation[param] = np.clip(variation[param] + delta, 1, 50)
                    elif param in ['chroma']:
                        variation[param] = np.clip(variation[param] + delta, 1, 40)
                    elif param in ['scan']:
                        variation[param] = np.clip(variation[param] + delta, 60, 180)
                    elif param in ['noise']:
                        variation[param] = np.clip(variation[param] + delta, 0.02, 0.15)
                    elif param in ['energy', 'pulse']:
                        variation[param] = np.clip(variation[param] + delta, 0.1, 0.3)
            
            score, _ = self.evaluate_preset(preset_name, variation)
            
            if score > best_score:
                best_score = score
                best_params = variation.copy()
                print(f"New best score: {best_score:.3f}")
                print(f"Parameters: {best_params}")
        
        return best_score, best_params

    def analyze_all_presets(self):
        """Analyze and optimize all presets"""
        print("Starting preset analysis...")
        
        # First, evaluate all original presets
        print("\nEvaluating original presets:")
        for preset_name, params in EFFECT_PRESETS.items():
            score, _ = self.evaluate_preset(preset_name, params)
            self.preset_scores[preset_name] = score
            print(f"{preset_name}: {score:.3f}")
        
        # Sort presets by score
        sorted_presets = sorted(self.preset_scores.items(), 
                              key=lambda x: x[1], 
                              reverse=True)
        
        print("\nPreset rankings:")
        for preset_name, score in sorted_presets:
            print(f"{preset_name}: {score:.3f}")
        
        # Optimize each preset
        print("\nOptimizing presets...")
        for preset_name, params in EFFECT_PRESETS.items():
            best_score, best_params = self.optimize_preset(preset_name, params)
            self.optimized_presets[preset_name] = {
                'original_score': self.preset_scores[preset_name],
                'optimized_score': best_score,
                'original_params': params,
                'optimized_params': best_params
            }
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save analysis results"""
        results_path = self.output_dir / 'preset_analysis.txt'
        with open(results_path, 'w') as f:
            f.write("Preset Analysis Results\n")
            f.write("======================\n\n")
            
            for preset_name, data in self.optimized_presets.items():
                f.write(f"\n{preset_name}\n")
                f.write("-" * len(preset_name) + "\n")
                f.write(f"Original score: {data['original_score']:.3f}\n")
                f.write(f"Optimized score: {data['optimized_score']:.3f}\n")
                f.write(f"Improvement: {(data['optimized_score'] - data['original_score']):.3f}\n")
                f.write("\nOriginal parameters:\n")
                for k, v in data['original_params'].items():
                    f.write(f"  {k}: {v}\n")
                f.write("\nOptimized parameters:\n")
                for k, v in data['optimized_params'].items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")

if __name__ == "__main__":
    optimizer = PresetOptimizer()
    optimizer.analyze_all_presets()