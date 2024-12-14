from pathlib import Path
import numpy as np
from PIL import Image, ImageStat, ImageFilter
from tetimi import ImageProcessor
import time

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy import stats

class ParameterSearch:
    def __init__(self, test_image_path="input.png", output_dir="parameter_search"):
        self.test_image_path = Path(test_image_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)

        # RGB colors as list for proper random selection
        self.rgb_colors = [
            (20, 235, 215),   # cyberpunk
            (235, 100, 235),  # vaporwave
            (235, 45, 225),   # tetsuo
            (185, 25, 235),   # psychic
            (235, 25, 65),    # akira
            (225, 24, 42),    # tetsuo rage
            (65, 215, 95),    # retro
            (25, 225, 95),    # matrix
            (235, 35, 85),    # neo tokyo
        ]
        
        # Parameter ranges adjusted to match tetimi.py constraints
        self.param_ranges = {
            'alpha': [120, 140, 160, 180, 200],
            'glitch': [1, 3, 5, 8, 10, 15, 20],  # Must be between 1-50
            'chroma': [1, 3, 5, 8, 10, 15, 20],  # Must be between 1-40
            'noise': [0.02, 0.03, 0.04, 0.05, 0.06],
            'scan': [60, 90, 120, 150],
            'energy': [0.1, 0.2, 0.3],
            'pulse': [0.1, 0.2, 0.3],
            'silkscreen': [True, False]
        }
        
        # Effect priorities (higher priority = more likely to be selected)
        self.effect_priorities = {
            'rgb': 1,
            'glitch': 1,
            'chroma': 1,
            'noise': 1,
            'silkscreen': 1,
            'scan': 2,
            'alpha': 2,
            'energy': 3,
            'pulse': 3
        }
        
        self.results_file = self.output_dir / 'results.txt'
        self.high_scores_file = self.output_dir / 'high_scores.txt'

    def evaluate_combination(self, params, save_all=False):
        """Evaluate a parameter combination"""
        try:
            silkscreen = params.pop('silkscreen', False)
            processor = ImageProcessor(self.test_image_path, silkscreen=silkscreen)
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
            
            # Basic image analysis
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
            
            # Add silkscreen back to params
            params['silkscreen'] = silkscreen
            
            
            
            
            # Base image qualities
            brightness = sum(means) / (3 * 255.0)  # 0-1 scale
            contrast = sum(stddevs) / (3 * 255.0)  # 0-1 scale
            
            # Color assessment
            channel_separation = np.std(means) / 255.0
            color_balance = 1.0 - abs(0.5 - brightness)  # Prefer midtone images
            
            # Detail preservation
            edges = rgb_result.filter(ImageFilter.FIND_EDGES)
            edge_score = ImageStat.Stat(edges).mean[0] / 255.0
            
            # Effect-specific scoring
            effect_scores = []
            if 'scan' in params:
                # Detect regular patterns for scan lines
                vertical_variance = np.var(np.array(rgb_result)[:,:,0], axis=0).mean()
                effect_scores.append(min(vertical_variance / 1000.0, 1.0))
            
            if 'glitch' in params:
                # Look for horizontal discontinuities
                horizontal_variance = np.var(np.array(rgb_result)[:,:,0], axis=1).mean()
                effect_scores.append(min(horizontal_variance / 1000.0, 1.0))
            
            # Combine scores with appropriate weights
            score = (
                channel_separation * 0.2 +  # Color separation
                color_balance * 0.2 +       # Overall color balance
                edge_score * 0.2 +          # Edge detection
                contrast * 0.2 +            # Contrast
                (sum(effect_scores) / max(len(effect_scores), 1)) * 0.2  # Effect-specific scores
            )
            
            # Penalize extreme cases
            if brightness < 0.1 or brightness > 0.9:  # Too dark or too bright
                score *= 0.5
            if contrast < 0.1:  # Too flat
                score *= 0.5
            
            return score, result
            
        except Exception as e:
            print(f"Error evaluating combination: {str(e)}")
            return 0.0, None

    def format_params(self, params):
        """Format parameters for logging"""
        parts = []
        for k, v in sorted(params.items()):
            if isinstance(v, tuple):
                v_str = f"({','.join(str(x) for x in v)})"
            else:
                v_str = str(v)
            parts.append(f"{k}={v_str}")
        return ", ".join(parts)

    def save_image(self, result, params, score, prefix=""):
        """Save an image with its parameters in the filename"""
        if result:
            # Create a filename from parameters
            param_str = '_'.join(f"{k}{v}" for k, v in sorted(params.items()) 
                               if k not in ['silkscreen', 'rgb', 'alpha'])
            if len(param_str) > 100:  # Truncate if too long
                param_str = param_str[:100]
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}{timestamp}_{score:.3f}_{param_str}.png"
            
            # Clean filename of invalid characters
            filename = "".join(c for c in filename if c.isalnum() or c in "._-")
            
            image_path = self.images_dir / filename
            try:
                result.save(image_path)
                print(f"Saved image to {image_path}")
                return image_path
            except Exception as e:
                print(f"Error saving image: {str(e)}")
        return None

    def log_result(self, params, score, image_path=None):
        """Log a single result"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_params = self.format_params(params)
        log_line = f"[{timestamp}] Score: {score:.3f} | {formatted_params}"
        if image_path:
            log_line += f" | Image: {image_path.name}"
        log_line += "\n"
        
        # Always log to main results
        with open(self.results_file, 'a') as f:
            f.write(log_line)
        
        # Log high scores separately
        if score > 0.65:
            with open(self.high_scores_file, 'a') as f:
                f.write(log_line)

    def search(self, num_combinations=100, min_effects=2, max_effects=4, save_all=True):
        """Search through parameter combinations"""
        print(f"Starting parameter search with {num_combinations} combinations...")
        
        # Initialize files with headers
        for file in [self.results_file, self.high_scores_file]:
            if not file.exists():
                with open(file, 'w') as f:
                    f.write("# Parameter Search Results\n")
                    f.write("# Format: [timestamp] Score: score | param1=value1, param2=value2, ...\n\n")
        
        for i in range(num_combinations):
            # Select effects based on priorities
            available_effects = list(self.effect_priorities.keys())
            weights = [1.0/self.effect_priorities[effect] for effect in available_effects]
            weights = np.array(weights) / sum(weights)
            
            num_effects = np.random.randint(min_effects, max_effects + 1)
            selected_effects = np.random.choice(
                available_effects,
                size=min(num_effects, len(available_effects)),
                replace=False,
                p=weights
            )
            
            # Generate parameters
            params = {}
            for effect in selected_effects:
                if effect == 'rgb':
                    params[effect] = self.rgb_colors[np.random.randint(len(self.rgb_colors))]
                else:
                    params[effect] = np.random.choice(self.param_ranges[effect])
            
            # Evaluate combination
            score, result = self.evaluate_combination(params)
            
            # Save image based on conditions
            image_path = None
            if result:
                if save_all or score > 0.65:
                    prefix = "high_score_" if score > 0.65 else ""
                    image_path = self.save_image(result, params, score, prefix)
            
            # Log result
            self.log_result(params, score, image_path)
            
            if score > 0.65:
                print(f"\nHigh score found! Score: {score:.3f}")
                print(f"Parameters: {self.format_params(params)}")
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{num_combinations} combinations...")





    def compute_scc(img1, img2):
        """Compute Spatial Correlation Coefficient"""
        return stats.pearsonr(img1.flatten(), img2.flatten())[0]

    def compute_sam(img1, img2):
        """Compute Spectral Angle Mapper"""
        dot_product = np.sum(img1 * img2, axis=2)
        norm_1 = np.sqrt(np.sum(img1 ** 2, axis=2))
        norm_2 = np.sqrt(np.sum(img2 ** 2, axis=2))
        return np.arccos(dot_product / (norm_1 * norm_2))

    def evaluate_quality(result, original):
        """Evaluate image quality using multiple metrics"""
        try:
            # Convert to arrays
            result_array = np.array(result)
            original_array = np.array(original)
            
            # Compute metrics
            ssim_score = ssim(original_array, result_array, channel_axis=2)
            psnr_score = psnr(original_array, result_array)
            scc_score = compute_scc(original_array, result_array)
            sam_score = np.mean(compute_sam(original_array, result_array))
            
            # Normalize scores
            norm_psnr = np.clip(psnr_score / 50.0, 0, 1)  # Typical PSNR range 0-50
            norm_sam = 1 - (sam_score / np.pi)  # SAM range 0-π
            
            # Weighted combination
            final_score = (
                ssim_score * 0.4 +    # Structure preservation
                norm_psnr * 0.2 +     # Signal quality
                scc_score * 0.2 +     # Spatial correlation
                norm_sam * 0.2        # Spectral similarity
            )
            
            return final_score
            
        except Exception as e:
            print(f"Error in quality evaluation: {str(e)}")
            return 0.0
if __name__ == "__main__":
    print("Starting parameter search...")
    searcher = ParameterSearch()
    # Set save_all=True to save all images, False to save only high scores
    searcher.search(num_combinations=200, min_effects=2, max_effects=4, save_all=True)