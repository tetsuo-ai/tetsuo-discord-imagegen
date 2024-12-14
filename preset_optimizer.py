from pathlib import Path
import numpy as np
from PIL import Image, ImageStat, ImageFilter, ImageDraw
from tetimi import ImageProcessor, EFFECT_PRESETS, EFFECT_ORDER
import itertools
import time
import json
from datetime import datetime
import csv
from copy import deepcopy

class DeepPresetOptimizer:
    def __init__(self, test_image_path="input.png", output_dir="preset_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.images_dir = self.output_dir / "images"
        self.history_dir = self.output_dir / "history"
        self.generations_dir = self.output_dir / "generations"
        
        for dir_path in [self.images_dir, self.history_dir, self.generations_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Create or verify test image
        self.test_image_path = Path(test_image_path)
        self.ensure_test_image()
        
        # Parameter variation ranges
        self.variation_ranges = {
            'alpha': (-30, 30),
            'glitch': (-3, 3),
            'chroma': (-3, 3),
            'scan': (-30, 30),
            'noise': (-0.02, 0.02),
            'energy': (-0.1, 0.1),
            'pulse': (-0.1, 0.1)
        }
        
        # Store results and history
        self.preset_scores = {}
        self.optimized_presets = {}
        self.generation_history = {}
        self.best_variations = {}
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def ensure_test_image(self):
        """Create test image if it doesn't exist"""
        if not self.test_image_path.exists():
            print(f"Creating test image at {self.test_image_path}")
            # Create a more interesting test image
            width, height = 800, 600
            image = Image.new('RGB', (width, height), 'black')
            draw = ImageDraw.Draw(image)
            
            # Add some shapes for testing effects
            # Background gradient
            for y in range(height):
                color = int(255 * y / height)
                draw.line([(0, y), (width, y)], fill=(color, color, color))
            
            # Add geometric shapes
            draw.rectangle([100, 100, 300, 300], fill='white')
            draw.ellipse([400, 200, 600, 400], fill='gray')
            draw.polygon([(350, 50), (450, 150), (250, 150)], fill='white')
            
            # Add some lines for testing scan lines
            for y in range(0, height, 20):
                draw.line([(0, y), (width, y)], fill='white', width=2)
            
            image.save(self.test_image_path)
            print(f"Created test image with dimensions {width}x{height}")

    def evaluate_preset(self, preset_name, params, save_image=False):
        """Enhanced preset evaluation with more detailed metrics"""
        try:
            if not self.test_image_path.exists():
                self.ensure_test_image()
            
            processor = ImageProcessor(str(self.test_image_path))
            result = processor.base_image.convert('RGBA')
            
            # Apply effects in order
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
            
            # Enhanced analysis
            rgb_result = result.convert('RGB')
            stats = ImageStat.Stat(rgb_result)
            means = stats.mean
            stddevs = stats.stddev
            
            # Detailed metrics
            brightness = sum(means) / (3 * 255.0)
            contrast = sum(stddevs) / (3 * 255.0)
            channel_separation = np.std(means) / 255.0
            
            # Edge analysis
            edges = rgb_result.filter(ImageFilter.FIND_EDGES)
            edge_score = ImageStat.Stat(edges).mean[0] / 255.0
            
            # Calculate histogram entropy for complexity
            hist = rgb_result.histogram()
            hist_norm = [h/sum(hist) for h in hist if h != 0]
            entropy = -sum(p * np.log2(p) for p in hist_norm)
            complexity = entropy / 16
            
            # Penalties
            washed_out_penalty = 1.0 if all(m > 240 for m in means) else 0.0
            too_dark_penalty = 1.0 if all(m < 15 for m in means) else 0.0
            flat_penalty = 1.0 if all(s < 10 for s in stddevs) else 0.0
            
            # Comprehensive scoring
            score = (
                0.25 * contrast +
                0.20 * channel_separation +
                0.25 * edge_score +
                0.15 * complexity +
                0.15 * (1 - abs(0.5 - brightness))
            ) * (1 - max(washed_out_penalty, too_dark_penalty, flat_penalty))
            
            # Save high-scoring variations
            if save_image and score > 0.5:
                image_path = self.images_dir / f"{preset_name}_{self.timestamp}_{score:.3f}.png"
                result.save(image_path)
            
            metrics = {
                'score': score,
                'brightness': brightness,
                'contrast': contrast,
                'channel_separation': channel_separation,
                'edge_detail': edge_score,
                'complexity': complexity,
                'penalties': {
                    'washed_out': washed_out_penalty,
                    'too_dark': too_dark_penalty,
                    'flat': flat_penalty
                }
            }
            
            return metrics, result
            
        except Exception as e:
            print(f"Error evaluating {preset_name}: {str(e)}")
            return None, None

    def optimize_preset(self, preset_name, preset_params, generations=100, population_size=20):
        """Deep optimization with generational history"""
        # Initialize with empty lists to avoid max() error on empty sequence
        generation_data = []
        best_score = 0.0
        best_params = preset_params.copy()
        
        # Evaluate original preset
        original_metrics, _ = self.evaluate_preset(preset_name, preset_params)
        if original_metrics:
            original_score = original_metrics['score']
            population = [(preset_params, original_score)]
        else:
            original_score = 0.0
            return best_score, best_params, generation_data
        
        print(f"\nOptimizing {preset_name}...")
        print(f"Original parameters: {preset_params}")
        print(f"Original score: {original_score:.3f}")
        
        for gen in range(generations):
            gen_variations = []
            
            for _ in range(population_size):
                try:
                    parent = max(population, key=lambda x: x[1])[0]
                    variation = deepcopy(parent)
                    
                    num_params = np.random.randint(1, 4)
                    params_to_vary = np.random.choice(
                        list(self.variation_ranges.keys()),
                        size=min(num_params, len(self.variation_ranges)),
                        replace=False
                    )
                    
                    for param in params_to_vary:
                        if param in variation:
                            min_val, max_val = self.variation_ranges[param]
                            delta = np.random.uniform(min_val, max_val)
                            
                            if param == 'alpha':
                                variation[param] = np.clip(variation[param] + delta, 120, 220)
                            elif param == 'glitch':
                                variation[param] = np.clip(variation[param] + delta, 1, 50)
                            elif param == 'chroma':
                                variation[param] = np.clip(variation[param] + delta, 1, 40)
                            elif param == 'scan':
                                variation[param] = np.clip(variation[param] + delta, 60, 180)
                            elif param == 'noise':
                                variation[param] = np.clip(variation[param] + delta, 0.02, 0.15)
                            elif param in ['energy', 'pulse']:
                                variation[param] = np.clip(variation[param] + delta, 0.1, 0.3)
                    
                    metrics, _ = self.evaluate_preset(preset_name, variation, save_image=True)
                    if metrics:
                        score = metrics['score']
                        gen_variations.append((variation, score, metrics))
                
                except Exception as e:
                    print(f"Error in generation {gen + 1}: {str(e)}")
                    continue
            
            if gen_variations:
                population = sorted(gen_variations, key=lambda x: x[1], reverse=True)[:population_size]
                
                current_best = max(population, key=lambda x: x[1])
                if current_best[1] > best_score:
                    best_score = current_best[1]
                    best_params = current_best[0]
                    print(f"Generation {gen + 1}: New best score: {best_score:.3f}")
                    print(f"Parameters: {best_params}")
                
                generation_data.append({
                    'generation': gen + 1,
                    'best_score': best_score,
                    'best_params': best_params,
                    'population': [(p[0], p[1], p[2]) for p in population]
                })
                
                self.save_generation_data(preset_name, gen + 1, generation_data[-1])
        
        return best_score, best_params, generation_data

    def save_generation_data(self, preset_name, generation, data):
        """Save detailed generation data"""
        gen_file = self.generations_dir / f"{preset_name}_gen_{generation}.json"
        with open(gen_file, 'w') as f:
            json.dump({
                'generation': generation,
                'best_score': data['best_score'],
                'best_params': data['best_params'],
                'population_scores': [p[1] for p in data['population']],
                'metrics': [p[2] for p in data['population']]
            }, f, indent=2)

    def analyze_all_presets(self):
        """Analyze and optimize all presets with full history"""
        print("Starting deep preset analysis...")
        
        print("\nEvaluating original presets:")
        for preset_name, params in EFFECT_PRESETS.items():
            metrics, _ = self.evaluate_preset(preset_name, params)
            if metrics:
                self.preset_scores[preset_name] = metrics['score']
                print(f"{preset_name}: {metrics['score']:.3f}")
        
        sorted_presets = sorted(self.preset_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("\nPreset rankings:")
        for preset_name, score in sorted_presets:
            print(f"{preset_name}: {score:.3f}")
        
        print("\nPerforming deep optimization...")
        for preset_name, params in EFFECT_PRESETS.items():
            best_score, best_params, history = self.optimize_preset(preset_name, params)
            
            self.optimized_presets[preset_name] = {
                'original_score': self.preset_scores.get(preset_name, 0.0),
                'optimized_score': best_score,
                'original_params': params,
                'optimized_params': best_params
            }
            
            self.generation_history[preset_name] = history
        
        self.save_results()

    def save_results(self):
        """Save comprehensive analysis results"""
        results_path = self.output_dir / f'preset_analysis_{self.timestamp}.txt'
        with open(results_path, 'w') as f:
            f.write("Deep Preset Analysis Results\n")
            f.write("==========================\n\n")
            
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
        
        history_path = self.history_dir / f'optimization_history_{self.timestamp}.csv'
        with open(history_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Preset', 'Generation', 'Best Score', 'Avg Score', 'Parameters'])
            
            for preset_name, history in self.generation_history.items():
                for gen_data in history:
                    avg_score = np.mean([p[1] for p in gen_data['population']])
                    writer.writerow([
                        preset_name,
                        gen_data['generation'],
                        gen_data['best_score'],
                        avg_score,
                        str(gen_data['best_params'])
                    ])

if __name__ == "__main__":
    optimizer = DeepPresetOptimizer()
    optimizer.analyze_all_presets()