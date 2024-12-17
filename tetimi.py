from PIL import (
    Image, 
    ImageFont, 
    ImageDraw, 
    ImageOps, 
    ImageEnhance, 
    ImageFilter, 
    ImageStat,
    ImageChops 
)
import numpy as np
from pathlib import Path
import random
import colorsys
import math
from typing import Optional, Tuple, Dict, Any, List


# Effect order and presets
EFFECT_ORDER = ['rgb', 'color', 'glitch', 'chroma', 'scan', 'noise', 'energy', 'pulse']
EFFECT_PRESETS = {
    'cyberpunk': {
        'rgb': (20, 235, 215),
        'rgbalpha': 75,
        'color': (20, 235, 215),
        'coloralpha': 50,
        'chroma': 5,
        'glitch': 3,
        'scan': 80,
        'noise': 0.03
    },
    'vaporwave': {
        'color': (235, 100, 235),
        'coloralpha': 85,
        'chroma': 8,
        'scan': 100,
        'noise': 0.02,
        'rgbalpha': 65
    },
    'glitch_art': {
        'color': (235, 45, 75),
        'coloralpha': 90,
        'glitch': 12,
        'chroma': 15,
        'scan': 120,
        'noise': 0.08,
        'rgbalpha': 60
    },
    'retro': {
        'rgb': (65, 215, 95),
        'rgbalpha': 50,
        'color': (65, 215, 95),
        'coloralpha': 50,
        'scan': 90,
        'noise': 0.03
    },
    'matrix': {
        'color': (25, 225, 95),
        'coloralpha': 95,
        'scan': 70,
        'glitch': 2,
        'noise': 0.02,
        'chroma': 3,
        'rgbalpha': 45
    },
    'synthwave': {
        'color': (225, 45, 235),
        'coloralpha': 90,
        'chroma': 7,
        'scan': 150,
        'noise': 0.02,
        'rgbalpha': 40
    },
    'akira': {
        'color': (235, 25, 65),
        'coloralpha': 95,
        'chroma': 8,
        'glitch': 6,
        'scan': 180,
        'noise': 0.03,
        'rgbalpha': 35
    },
    'tetsuo': {
        'color': (235, 45, 225),
        'coloralpha': 100,
        'chroma': 10,
        'glitch': 6,
        'scan': 160,
        'noise': 0.05,
        'pulse': 0.15,
        'rgbalpha': 30
    },
    'neo_tokyo': {
        'rgb': (235, 35, 85),
        'rgbalpha': 25,
        'color': (235, 35, 85),
        'coloralpha': 85,
        'chroma': 12,
        'glitch': 8,
        'scan': 140,
        'noise': 0.04,
        'pulse': 0.1
    },
    'psychic': {
        'color': (185, 25, 235),
        'coloralpha': 95,
        'chroma': 15,
        'glitch': 5,
        'scan': 130,
        'noise': 0.03,
        'energy': 0.2,
        'rgbalpha': 85
    },
    'tetsuo_rage': {
        'color': (225, 24, 42),
        'coloralpha': 120,
        'chroma': 20,
        'glitch': 25,
        'scan': 50,
        'noise': 0.15,
        'energy': 0.2,
        'pulse': 0.2,
        'rgbalpha': 90
    }
}

class ImageAnalyzer:
    @staticmethod
    def analyze_image(image: Image.Image) -> Dict[str, float]:
        """Comprehensive image analysis for adaptive processing"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        stat = ImageStat.Stat(image)
        brightness = sum(stat.mean) / (3 * 255.0)
        contrast = sum(stat.stddev) / (3 * 255.0)
        
        r, g, b = stat.mean
        color_variance = np.std([r, g, b])
        
        edges = image.filter(ImageFilter.FIND_EDGES)
        edge_stat = ImageStat.Stat(edges)
        complexity = sum(edge_stat.mean) / (3 * 255.0)
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'color_variance': color_variance,
            'complexity': complexity
        }
    
    @staticmethod
    def get_adaptive_params(analysis: Dict[str, float], 
                          user_params: Optional[Dict[str, Any]] = None,
                          preset_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate adaptive parameters based on image analysis"""
        params = {}
        requested_effects = set()
        
        if user_params:
            requested_effects.update(user_params.keys())
        if preset_params:
            requested_effects.update(preset_params.keys())
        
        if 'alpha' in requested_effects:
            params['alpha'] = int(255 * (1.0 - (analysis['brightness'] * 0.7 + 
                                               analysis['contrast'] * 0.3)))
        
        if 'glitch' in requested_effects:
            params['glitch'] = int(20 * (1.0 - (analysis['complexity'] * 0.8)))
        
        if 'chroma' in requested_effects:
            params['chroma'] = int(20 * (1.0 - (analysis['color_variance']/255 * 0.9)))
        
        if 'noise' in requested_effects:
            params['noise'] = float(analysis['contrast'] * 0.5 + 
                                  analysis['brightness'] * 0.5)
        
        if 'scan' in requested_effects:
            params['scan'] = int(10 * (1.0 - analysis['complexity']))
        
        return params

def offset_channel(image: Image.Image, offset_x: int, offset_y: int) -> Image.Image:
    """Enhanced channel offset with smoother transitions"""
    width, height = image.size
    offset_image = Image.new(image.mode, (width, height), 0)
    
    if offset_x > 0:
        left_part = image.crop((width - offset_x, 0, width, height))
        main_part = image.crop((0, 0, width - offset_x, height))
        offset_image.paste(left_part, (0, 0))
        offset_image.paste(main_part, (offset_x, 0))
    elif offset_x < 0:
        right_part = image.crop((0, 0, -offset_x, height))
        main_part = image.crop((-offset_x, 0, width, height))
        offset_image.paste(right_part, (width + offset_x, 0))
        offset_image.paste(main_part, (0, 0))
    else:
        offset_image = image.copy()
    
    offset_image = offset_image.filter(ImageFilter.GaussianBlur(0.5))
    return offset_image

class ImageProcessor:
    def __init__(self, image_path: str, points: bool = False):
        """Initialize processor with points effect option
        
        Args:
            image_path: Path to input image
            points: Whether to apply points effect
        """
        if not Path(image_path).exists():
            raise ValueError(f"Image file not found: {image_path}")
            
        self.base_image = Image.open(image_path).convert('RGBA')
        if points:
            self.base_image = self.apply_points_effect()
                
        self.analyzer = ImageAnalyzer()
        self.analysis = self.analyzer.analyze_image(self.base_image)
        self.adaptive_params = {}

    def apply_points_effect(self, colors: Optional[List[str]] = None, 
                          dot_size: int = 6, 
                          registration_offset: int = 8) -> Image.Image:
        """Apply points effect to image"""
        if colors is None:
            colors = ['#E62020', '#20B020', '#2020E6', '#D4D420']
        
        image = self.base_image.convert('L')
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2)
        
        result = Image.new('RGBA', image.size, (255, 255, 255, 255))
        
        for i, color in enumerate(colors):
            r, g, b = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            halftone = Image.new('RGBA', image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(halftone)
            
            density_factor = 0.2 if color.startswith('#D4') else 0.9 if color.startswith('#E6') or color.startswith('#20B') else 0.7
            
            for y in range(0, image.size[1], dot_size):
                for x in range(0, image.size[0], dot_size):
                    box = (x, y, min(x + dot_size, image.size[0]), 
                          min(y + dot_size, image.size[1]))
                    region = image.crop(box)
                    average = ImageStat.Stat(region).mean[0]
                    
                    brightness_factor = ((255 - average) / 255.0) ** 0.8
                    adjusted_brightness = brightness_factor * density_factor
                    
                    dot_radius = int(adjusted_brightness * dot_size * 0.5)
                    
                    if dot_radius > 0:
                        offset_x = random.randint(-1, 1) * 0.5
                        offset_y = random.randint(-1, 1) * 0.5
                        center = (x + dot_size//2 + offset_x, 
                                y + dot_size//2 + offset_y)
                        
                        opacity = int(255 * min(adjusted_brightness * 0.9, 1.0))
                        draw.ellipse([center[0] - dot_radius, 
                                    center[1] - dot_radius,
                                    center[0] + dot_radius, 
                                    center[1] + dot_radius], 
                                    fill=(r, g, b, opacity))
            
            offset_x = int((i - len(colors)/2) * registration_offset * 0.8)
            offset_y = int((i - len(colors)/2) * registration_offset * 0.8)
            halftone = ImageChops.offset(halftone, offset_x, offset_y)
            
            result = Image.alpha_composite(result, halftone)
        
        return result

    def merge_params(self, user_params: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user parameters with adaptive parameters"""
        preset_name = user_params.get('preset')
        preset_params = EFFECT_PRESETS.get(preset_name, {})
        
        self.adaptive_params = self.analyzer.get_adaptive_params(
            self.analysis,
            user_params,
            preset_params
        )
        
        merged = self.adaptive_params.copy()
        
        if preset_params:
            merged.update(preset_params)
        
        merged.update(user_params)
        return merged

    def add_color_overlay(self, color: Tuple[int, int, int, int]) -> Image.Image:
        """Add subtle color overlay"""
        r, g, b, alpha = color
        blend_alpha = alpha / 255.0
        overlay = Image.new('RGBA', self.base_image.size, (r, g, b, int(alpha * 0.5)))
        result = Image.blend(self.base_image.convert('RGBA'), overlay, blend_alpha * 0.3)
        return result

    def colorize_non_white(self, r: int, g: int, b: int, alpha: int = 255) -> Image.Image:
        """Enhanced colorization with more nuanced color blending"""
        img_array = np.array(self.base_image.convert('RGBA'))
        glow = img_array.copy()
        glow[:, :] = [r, g, b, alpha]
        
        luminance = np.sum(img_array[:, :, :3] * [0.299, 0.587, 0.114], axis=2)
        non_white_mask = luminance < 240
        blend_factor = ((255 - luminance) / 255.0)[:, :, np.newaxis]
        blend_factor = np.clip(blend_factor * 1.5, 0, 1)
        
        result = img_array.copy()
        result[non_white_mask] = (
            (1 - blend_factor[non_white_mask]) * img_array[non_white_mask] +
            blend_factor[non_white_mask] * glow[non_white_mask]
        ).astype(np.uint8)
        
        return Image.fromarray(result)

    def apply_glitch_effect(self, intensity: int = 10) -> Image.Image:
        """Enhanced glitch effect with support for higher intensities"""
        if not isinstance(intensity, int) or not 1 <= intensity <= 50:
            raise ValueError("Glitch intensity must be an integer between 1 and 50")
            
        img_array = np.array(self.base_image)
        result = img_array.copy()
        
        iterations = int(intensity * 1.5)
        
        for _ in range(iterations):
            offset = np.random.randint(-20, 20)
            if offset == 0:
                continue
                
            if offset > 0:
                result[offset:, :] = img_array[:-offset, :]
                result[:offset, :] = img_array[-offset:, :]
            else:
                offset = abs(offset)
                result[:-offset, :] = img_array[offset:, :]
                result[-offset:, :] = img_array[:offset, :]
            
            channel = np.random.randint(0, 3)
            shift = np.random.randint(-10, 11)
            if shift != 0:
                temp = np.roll(result[:, :, channel], shift, axis=1)
                img_array[:, :, channel] = temp
        
        return Image.fromarray(result)

    def add_chromatic_aberration(self, offset):
        """
        Args:
            offset (int or tuple): Chromatic aberration offset
        """
        # If tuple, use numpy's linspace to generate range
        if isinstance(offset, tuple):
            start, end = offset
            # Convert to list of integers in range
            offset = np.linspace(start, end, num=10).astype(int)
        
        # Validate offset
        if isinstance(offset, (list, np.ndarray)):
            offset = [max(1, min(40, val)) for val in offset]
        elif not isinstance(offset, int) or not 1 <= offset <= 40:
            raise ValueError("Chromatic aberration offset must be between 1 and 40")
            
        # Split into channels
        r, g, b, a = self.base_image.split()
        
        # Calculate exponential offsets for more natural distortion
        r_offset = int(-offset * (1.0 + math.log(offset/10 + 1, 2)) * 0.8)
        b_offset = int(offset * (1.0 + math.log(offset/10 + 1, 2)) * 0.8)
        g_offset = int(offset * math.log(offset/20 + 1, 2) * 0.3)
        
        # Apply graduated blur based on offset distance
        blur_amount = 0.3 + (offset / 40) * 0.7
        
        # Process each channel with variable blur
        r = offset_channel(r, r_offset, 0)
        r = r.filter(ImageFilter.GaussianBlur(radius=blur_amount))
        
        b = offset_channel(b, b_offset, 0)
        b = b.filter(ImageFilter.GaussianBlur(radius=blur_amount))
        
        g = offset_channel(g, g_offset, 0)
        g = g.filter(ImageFilter.GaussianBlur(radius=blur_amount * 0.5))
        
        # Merge with slight alpha adjustment for edge cases
        result = Image.merge('RGBA', (r, g, b, a))
        
        # Enhance edge contrast slightly
        enhancer = ImageEnhance.Contrast(result)
        result = enhancer.enhance(1.1)
        
        return result
    
    def add_scan_lines(self, gap: int = 2, alpha: int = 128) -> Image.Image:
        """Enhanced scan lines with variable intensity and subtle glow effect"""
        width, height = self.base_image.size
        scan_lines = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(scan_lines)
        
        for y in range(0, height, gap):
            intensity = int(alpha * (0.7 + 0.3 * random.random()))
            draw.line([(0, y), (width, y)], fill=(0, 0, 0, intensity))
            
            if y > 0:
                draw.line([(0, y-1), (width, y-1)], 
                         fill=(0, 0, 0, intensity//2))
        
        scan_lines = scan_lines.filter(ImageFilter.GaussianBlur(0.5))
        return Image.alpha_composite(self.base_image.convert('RGBA'), scan_lines)
    
    def add_noise(self, intensity: float = 0.1) -> Image.Image:
        """Enhanced noise effect with color preservation"""
        if not isinstance(intensity, (int, float)) or not 0 <= intensity <= 1:
            raise ValueError("Noise intensity must be between 0 and 1")
            
        img_array = np.array(self.base_image)
        
        # Generate colored noise
        noise = np.random.normal(0, intensity * 255, img_array.shape)
        
        # Preserve color relationships
        noise_mask = np.random.random(img_array.shape) > 0.5
        noisy_image = img_array + (noise * noise_mask)
        
        np.clip(noisy_image, 0, 255, out=noisy_image)
        return Image.fromarray(noisy_image.astype('uint8'))

    def apply_energy_effect(self, intensity: float = 0.8) -> Image.Image:
        """Refined energy effect with intelligent line placement and glow
        
        Args:
            intensity (float): Effect intensity (0-2)
            
        Returns:
            PIL.Image: Processed image with energy effect
        """
        if not 0 <= intensity <= 2:
            raise ValueError("Energy intensity must be between 0 and 2")
            
        base = self.base_image.convert('RGBA')
        width, height = base.size
        
        # Create edge map for intelligent line placement
        edges = base.filter(ImageFilter.FIND_EDGES)
        edge_data = np.array(edges.convert('L'))
        
        # Create energy layer
        energy = Image.new('RGBA', base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(energy)
        
        # Calculate dynamic number of lines based on image size and intensity
        base_lines = int(min(width, height) * 0.15)
        num_lines = int(base_lines * intensity)
        
        # Track line positions for spacing
        line_positions = []
        
        # Generate lines with improved placement
        for _ in range(num_lines):
            # Find areas with strong edges
            edge_positions = np.where(edge_data > 50)
            if len(edge_positions[0]) > 0:
                # Randomly select from edge points
                idx = np.random.randint(len(edge_positions[0]))
                x1 = edge_positions[1][idx]
                y1 = edge_positions[0][idx]
            else:
                # Fallback to random position
                x1 = random.randint(0, width)
                y1 = random.randint(0, height)
            
            # Check spacing from existing lines
            if line_positions and any(abs(x1 - x) + abs(y1 - y) < 20 for x, y in line_positions):
                continue
                
            # Calculate dynamic line properties
            angle = random.uniform(0, 2 * math.pi)
            length = random.randint(int(30 * intensity), int(100 * intensity))
            
            # Generate end point
            x2 = x1 + int(length * math.cos(angle))
            y2 = y1 + int(length * math.sin(angle))
            
            # Generate color with controlled randomness
            hue = random.uniform(0.5, 0.7)  # Blue to purple range
            saturation = random.uniform(0.8, 1.0)
            value = random.uniform(0.8, 1.0)
            rgb = tuple(int(x * 255) for x in colorsys.hsv_to_rgb(hue, saturation, value))
            
            # Calculate alpha based on edge strength
            edge_strength = edge_data[y1, x1] / 255.0
            base_alpha = int(180 * intensity)
            alpha = int(base_alpha * (0.5 + 0.5 * edge_strength))
            color = rgb + (alpha,)
            
            # Draw line with dynamic width
            line_width = max(1, int(3 * intensity * (0.5 + 0.5 * edge_strength)))
            draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width)
            
            # Add glow effect
            glow_radius = int(line_width * 2)
            for r in range(glow_radius, 0, -1):
                glow_alpha = int(alpha * (r / glow_radius) * 0.3)
                glow_color = rgb + (glow_alpha,)
                draw.line([(x1, y1), (x2, y2)], fill=glow_color, width=line_width + r * 2)
            
            line_positions.append((x1, y1))
            if len(line_positions) > 10:
                line_positions.pop(0)
        
        # Apply graduated blur
        blur_radius = 1 + intensity
        energy = energy.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Blend with base image using overlay mode
        blend_factor = min(0.7, intensity * 0.4)
        result = Image.blend(base, Image.alpha_composite(base, energy), blend_factor)
        
        return result

    def apply_pulse_effect(self, intensity: float = 0.7) -> Image.Image:
        """Enhanced pulse effect with content-aware placement and dynamic sizing
        
        Args:
            intensity (float): Effect intensity (0-2)
            
        Returns:
            PIL.Image: Processed image with pulse effect
        """
        if not 0 <= intensity <= 2:
            raise ValueError("Pulse intensity must be between 0 and 2")
            
        base = self.base_image.convert('RGBA')
        width, height = base.size
        
        # Create analysis layers
        edges = base.filter(ImageFilter.FIND_EDGES)
        edge_data = np.array(edges.convert('L'))
        
        brightness_layer = base.convert('L')
        brightness_data = np.array(brightness_layer)
        
        # Create pulse layer
        pulse = Image.new('RGBA', base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(pulse)
        
        # Calculate number of pulses based on image size and intensity
        base_pulses = int(min(width, height) * 0.05)
        num_pulses = int(base_pulses * intensity)
        
        # Track pulse positions
        pulse_positions = []
        
        for _ in range(num_pulses):
            # Find dark areas with edges
            candidate_positions = np.where((brightness_data < 128) & (edge_data > 30))
            
            if len(candidate_positions[0]) > 0:
                # Select random position from candidates
                idx = np.random.randint(len(candidate_positions[0]))
                y = candidate_positions[0][idx]
                x = candidate_positions[1][idx]
            else:
                # Fallback to random position
                x = random.randint(0, width)
                y = random.randint(0, height)
                
            # Check spacing from existing pulses
            if pulse_positions and any(abs(x - px) + abs(y - py) < 40 for px, py in pulse_positions):
                continue
            
            # Calculate pulse properties based on local image data
            local_brightness = brightness_data[
                max(0, y-10):min(height, y+10),
                max(0, x-10):min(width, x+10)
            ].mean()
            
            local_edge = edge_data[
                max(0, y-10):min(height, y+10),
                max(0, x-10):min(width, x+10)
            ].mean()
            
            # Dynamic radius based on local properties
            base_radius = int(20 * (1 - local_brightness/255))
            variation = random.uniform(0.8, 1.2)
            radius = int(base_radius * variation * intensity)
            
            # Generate pulse with multiple layers
            num_layers = int(5 + intensity * 5)
            for layer in range(num_layers):
                progress = layer / num_layers
                current_radius = int(radius * (1 - progress))
                
                # Calculate alpha based on layer and local properties
                base_alpha = int(150 * intensity * (1 - local_brightness/255))
                layer_alpha = int(base_alpha * (1 - progress) * (0.5 + 0.5 * local_edge/255))
                
                # Generate colors with slight variation
                hue = random.uniform(0.55, 0.65)  # Blue range
                saturation = random.uniform(0.7, 0.9)
                value = random.uniform(0.8, 1.0)
                rgb = tuple(int(x * 255) for x in colorsys.hsv_to_rgb(hue, saturation, value))
                
                # Draw graduated pulse
                draw.ellipse(
                    [x - current_radius, y - current_radius,
                     x + current_radius, y + current_radius],
                    fill=rgb + (layer_alpha,)
                )
            
            pulse_positions.append((x, y))
            if len(pulse_positions) > 5:
                pulse_positions.pop(0)
        
        # Apply subtle blur
        blur_radius = 2 + intensity
        pulse = pulse.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Blend with base image
        blend_factor = min(0.6, intensity * 0.35)
        result = Image.blend(base, Image.alpha_composite(base, pulse), blend_factor)
        
        # Slight contrast enhancement
        enhancer = ImageEnhance.Contrast(result)
        result = enhancer.enhance(1.1)
        
        return result
    def apply_effect(self, effect_name: str, params: dict) -> Image.Image:
        """Apply named effect with parameters"""
        if effect_name == 'rgb':
            return self.colorize_non_white(*params['rgb'], params.get('rgbalpha', 255))
        elif effect_name == 'color':
            return self.add_color_overlay((*params['color'], params.get('coloralpha', 255)))
        elif effect_name == 'glitch':
            return self.apply_glitch_effect(params['glitch'])
        elif effect_name == 'chroma':
            return self.add_chromatic_aberration(params['chroma'])
        elif effect_name == 'scan':
            return self.add_scan_lines(params['scan'])
        elif effect_name == 'noise':
            return self.add_noise(params['noise'])
        elif effect_name == 'energy':
            return self.apply_energy_effect(params['energy'])
        elif effect_name == 'pulse':
            return self.apply_pulse_effect(params['pulse'])
        return self.base_image    
    def convertImageToAscii(self, cols: int = 80, scale: float = 0.43, moreLevels: bool = True) -> List[str]:
        """Convert image to ASCII art"""
        gscale1 = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
        gscale2 = '@%#*+=-:. '
        
        image = self.base_image.convert('L')
        W, H = image.size
        w = W/cols
        h = w/scale
        rows = int(H/h)
        
        if cols > W or rows > H:
            raise ValueError("Image too small for specified columns")

        aimg = []
        for j in range(rows):
            y1 = int(j*h)
            y2 = int((j+1)*h)
            if j == rows-1:
                y2 = H
                
            aimg.append("")
            for i in range(cols):
                x1 = int(i*w)
                x2 = int((i+1)*w)
                if i == cols-1:
                    x2 = W
                    
                img = image.crop((x1, y1, x2, y2))
                avg = int(np.array(img).mean())
                
                if moreLevels:
                    gsval = gscale1[int((avg*69)/255)]
                else:
                    gsval = gscale2[int((avg*9)/255)]
                
                aimg[j] += gsval
        
        return aimg