import discord
from discord.ext import commands
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
from io import BytesIO
from dotenv import load_dotenv
import os
import re
import random
import colorsys
import math
import time

# Load environment variables
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
IMAGES_FOLDER = 'images'
INPUT_IMAGE = 'input.png'


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
# Set up Discord intents
intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True
bot = commands.Bot(command_prefix='!', intents=intents, help_command=None)


class ImageAnalyzer:
    @staticmethod
    def analyze_image(image):
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
    def get_adaptive_params(analysis, user_params=None, preset_params=None):
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

def offset_channel(image, offset_x, offset_y):
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
    def __init__(self, image_path, silkscreen=False):
        """Initialize processor with optional dual image mode"""
        self.base_image = Image.open(image_path).convert('RGBA')
        if silkscreen:
            self.base_image = self.apply_silkscreen_effect()
        else:
            if not Path(image_path).exists():
                raise ValueError(f"Image file not found: {image_path}")
                
        self.analyzer = ImageAnalyzer()
        self.analysis = self.analyzer.analyze_image(self.base_image)
        self.adaptive_params = {}

    def apply_silkscreen_effect(self, colors=None, dot_size=5, registration_offset=2):
        if colors is None:
            colors = ['#E62020', '#20B020', '#2020E6', '#D4D420']
        
        image = self.base_image.convert('L')
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result = Image.new('RGBA', image.size, (255, 255, 255, 255))
        
        for i, color in enumerate(colors):
            r, g, b = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            halftone = Image.new('RGBA', image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(halftone)
            
            density_factor = 0.8 if color.startswith('#D4') else 1.0
            
            for y in range(0, image.size[1], dot_size):
                for x in range(0, image.size[0], dot_size):
                    box = (x, y, min(x + dot_size, image.size[0]), 
                          min(y + dot_size, image.size[1]))
                    region = image.crop(box)
                    average = ImageStat.Stat(region).mean[0]
                    
                    brightness_factor = ((255 - average) / 255.0) ** 0.8
                    adjusted_brightness = brightness_factor * density_factor
                    
                    dot_radius = int(adjusted_brightness * dot_size * 0.4)
                    
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
    
    def merge_params(self, user_params):
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
    
    def add_color_overlay(self, color):
        """Add subtle color overlay"""
        r, g, b, alpha = color
        
        # Normalize alpha to 0-1 range for blending
        blend_alpha = alpha / 255.0
        
        # Reduce color intensity to prevent over-coloration
        overlay = Image.new('RGBA', self.base_image.size, (r, g, b, int(alpha * 0.5)))
        
        result = Image.blend(self.base_image.convert('RGBA'), overlay, blend_alpha * 0.3)
        
        return result
        
    def apply_energy_effect(self, intensity=0.8):
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

    def apply_pulse_effect(self, intensity=0.7):
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

    def convertImageToAscii(self, cols=80, scale=0.43, moreLevels=True):
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
    
    def apply_glitch_effect(self, intensity=10):
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
    def colorize_non_white(self, r, g, b, alpha=255):
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
    
    def add_chromatic_aberration(self, offset=10):
        """Enhanced chromatic aberration with natural distortion falloff
        
        Args:
            offset (int): Base offset amount (1-40)
            
        Returns:
            PIL.Image: Processed image with chromatic aberration effect
        """
        if not isinstance(offset, int) or not 1 <= offset <= 40:
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
    
    def add_scan_lines(self, gap=2, alpha=128):
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
    
    def add_noise(self, intensity=0.1):
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

def parse_discord_args(args):
    result = {}
    args = ' '.join(args)
    
    # Alpha handling
    alpha_match = re.search(r'--alpha\s+(\d+)', args)
    
    if alpha_match:
        alpha = int(alpha_match.group(1))
        if 0 <= alpha <= 255:
            result['alpha'] = alpha
        else:
            raise ValueError("Alpha value must be between 0 and 255")
    else:
        result['alpha'] = 180  # Default alpha value if not specified
        
    if '--silkscreen' in args:
        result['silkscreen'] = True        
    if '--random' in args:
        images = list(Path(IMAGES_FOLDER).glob('*.*'))
        if not images:
            raise ValueError(f"No images found in {IMAGES_FOLDER}")
        result['image_path'] = str(random.choice(images))
    
    preset_match = re.search(r'--preset\s+(\w+)', args)
    if preset_match:
        preset_name = preset_match.group(1).lower()
        if preset_name in EFFECT_PRESETS:
            result.update(EFFECT_PRESETS[preset_name])
            return result
        raise ValueError(f"Unknown preset. Available presets: {', '.join(EFFECT_PRESETS.keys())}")
    
    match = re.search(r'--(rgb|color)\s+(\d+)\s+(\d+)\s+(\d+)(?:\s+--\1alpha\s+(\d+))?', args)
    if match:
        color_type = match.group(1)  # Either 'rgb' or 'color'
        r, g, b = map(int, match.groups()[1:4])
        alpha = int(match.group(5)) if match.group(5) else 255
        
        if all(0 <= x <= 255 for x in [r, g, b, alpha]):
            result[color_type] = (r, g, b)
            result[f'{color_type}alpha'] = alpha
        else:
            raise ValueError("Color/Alpha values must be between 0 and 255")
    
    # Other parameter handling
    params = {
        'glitch': (r'--glitch\s+(\d*\.?\d+)', lambda x: 1 <= float(x) <= 50, "Glitch intensity must be between 1 and 50"),
        'chroma': (r'--chroma\s+(\d*\.?\d+)', lambda x: 1 <= float(x) <= 40, "Chromatic aberration must be between 1 and 40"),
        'scan': (r'--scan\s+(\d*\.?\d+)', lambda x: 1 <= float(x) <= 200, "Scan line gap must be between 1 and 200"),
        'noise': (r'--noise\s+(\d*\.?\d+)', lambda x: 0 <= float(x) <= 2, "Noise intensity must be between 0 and 2"),
        'energy': (r'--energy\s+(\d*\.?\d+)', lambda x: 0 <= float(x) <= 2, "Energy intensity must be between 0 and 2"),
        'pulse': (r'--pulse\s+(\d*\.?\d+)', lambda x: 0 <= float(x) <= 2, "Pulse intensity must be between 0 and 2")
    }
    
    for param, (pattern, validator, error_msg) in params.items():
        match = re.search(pattern, args)
        if match:
            value = float(match.group(1)) if param in ['noise', 'energy', 'pulse'] else int(match.group(1))
            if validator(value):
                result[param] = value
            else:
                raise ValueError(error_msg)
    
    return result

@bot.event
async def on_ready():
    print(f'Image generation bot is online as {bot.user}')
    print(f"Using input image: {INPUT_IMAGE}")
    if not Path(INPUT_IMAGE).exists():
        print(f"Warning: Input image '{INPUT_IMAGE}' not found!")

@bot.event
async def on_reaction_add(reaction, user):
    if user != bot.user and str(reaction.emoji) == "🗑️":
        if reaction.message.author == bot.user:
            await reaction.message.delete()

@bot.command(name='image')
async def image_command(ctx, *args):
    try:
        params = parse_discord_args(args)
        silkscreen = params.pop('silkscreen', False)
        image_path = params.pop('image_path', INPUT_IMAGE)
        
        processor = ImageProcessor(image_path, silkscreen=silkscreen)
        
        if not Path(image_path).exists():
            await ctx.send(f"Error: Image '{image_path}' not found!")
            return
        
        if '--ascii' in args:
            result = processor.generate_discord_ascii()
            output = BytesIO()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            result.save(output, format='PNG', quality=95)
            output.seek(0)
            await ctx.send(file=discord.File(fp=output, filename=f'{timestamp}_tetimi_ascii.png'))
            return
            
        if not params:
            presets_list = '\n'.join([f"- {name}: {', '.join(f'{k}={v}' for k, v in effects.items())}" 
                                    for name, effects in EFFECT_PRESETS.items()])
            await ctx.send(f"No valid arguments provided. Use !image_help for full options")
            return

        params = processor.merge_params(params)
        result = processor.base_image.convert('RGBA')
        
        for effect in EFFECT_ORDER:
            if effect not in params:
                continue
            
            processor.base_image = result.copy()
            if effect == 'rgb':
                r, g, b = params['rgb']
                alpha = params.get('rgbalpha', 255)  # Use rgbalpha specifically
                result = processor.add_color_overlay((r, g, b, alpha))
            elif effect == 'color':
                r, g, b = params['color']  # Use color params
                alpha = params.get('coloralpha', 255)  # Use coloralpha
                result = processor.colorize_non_white(r, g, b, alpha)  # Pass alpha
            elif effect == 'energy' and 'energy' in params:
                result = processor.apply_energy_effect(params['energy'])
            elif effect == 'pulse' and 'pulse' in params:
                result = processor.apply_pulse_effect(params['pulse'])
            elif effect == 'glitch':
                glitch_result = processor.apply_glitch_effect(params['glitch'])
                if glitch_result.mode != 'RGBA':
                    glitch_result = glitch_result.convert('RGBA')
                result = glitch_result
            elif effect == 'chroma':
                result = processor.add_chromatic_aberration(params['chroma'])

            elif effect == 'scan':
                result = processor.add_scan_lines(params['scan'])
            elif effect == 'noise':
                processor.base_image = result.copy()
                noise_result = processor.add_noise(params['noise'])
                if noise_result.mode != 'RGBA':
                    noise_result = noise_result.convert('RGBA')
                result = noise_result

        output = BytesIO()
        result.save(output, format='PNG')
        output.seek(0)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_tetimi.png"
        
        message = await ctx.send(file=discord.File(fp=output, filename=filename))
        await message.add_reaction("🗑️")
        
    except Exception as e:
        await ctx.send(f"Error: {str(e)}")


@bot.command(name='ascii')
async def ascii_art(ctx, *args):
    try:
        image_path = INPUT_IMAGE
        upscale = False
        
        # Process arguments
        if 'random' in args:
            images = list(Path(IMAGES_FOLDER).glob('*.*'))
            if not images:
                await ctx.send(f"Error: No images found in {IMAGES_FOLDER}")
                return
            image_path = str(random.choice(images))
        
        if 'up' in args:
            upscale = True
            
        if not Path(image_path).exists():
            await ctx.send(f"Error: Image '{image_path}' not found!")
            return
                
        processor = ImageProcessor(image_path)
        
        if upscale:
            # 4x upscale with enhancements
            original_size = processor.base_image.size
            new_size = (original_size[0] * 4, original_size[1] * 4)
            
            # Convert and enhance
            img = processor.base_image.convert('RGB')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.3)
            
            # High quality upscale
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            processor.base_image = img
            
            # Use higher columns for upscaled image
            cols = 160
        else:
            cols = 80
        
        # Generate ASCII
        ascii_lines = processor.convertImageToAscii(cols=cols, scale=0.43, moreLevels=True)
        
        # Create output file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"{timestamp}_tetimi_ascii.txt"
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(f"// Image: {Path(image_path).name}\n")
            if upscale:
                f.write(f"// Upscaled 4x with enhancements\n")
                f.write(f"// Original size: {original_size[0]}x{original_size[1]}\n")
                f.write(f"// New size: {new_size[0]}x{new_size[1]}\n")
            f.write(f"// Columns: {cols}\n")
            f.write(f"// Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for row in ascii_lines:
                f.write(row + '\n')
                
        # Send file
        await ctx.send(
            content=f"ASCII art generated from {Path(image_path).name}" + 
                    (" (4x upscaled)" if upscale else ""),
            file=discord.File(output_filename)
        )
        
        # Cleanup
        try:
            os.remove(output_filename)
        except Exception as e:
            print(f"Warning: Could not remove temporary file {output_filename}: {e}")
        
    except Exception as e:
        await ctx.send(f"Error: {str(e)}")

@bot.command(name='ascii_help')
async def ascii_help(ctx):
    embed = discord.Embed(
        title="ASCII Art Generator Commands",
        description="Convert images to ASCII art",
        color=discord.Color.purple()
    )
    
    embed.add_field(
        name="Basic Usage",
        value="!ascii - Convert default image\n" +
              "!ascii up - 4x upscale with enhancements\n" +
              "!ascii random - Use random image\n" +
              "!ascii up random - Random image with upscale",
        inline=False
    )
    
    await ctx.send(embed=embed)


@bot.command(name='image_help')
async def image_help(ctx):
    embed = discord.Embed(
        title="Image Generation Bot Commands",
        description="Image effects generator",
        color=discord.Color.purple()
    )
    
    embed.add_field(
        name="Basic Usage",
        value="!image [options] - Process default image\nimage --random - Process random Akira image",
        inline=False
    )
    
    embed.add_field(
        name="Presets",
        value="--preset [name]\nAvailable: cyberpunk, vaporwave, glitch_art, retro, matrix, synthwave, akira, tetsuo, neo_tokyo, psychic, tetsuo_rage",
        inline=False
    )
    
    embed.add_field(
        name="Effect Options",
        value="--rgb [r] [g] [b] --alpha [0-255]\n--glitch [1-50]\n--chroma [1-40]\n--scan [1-20]\n--noise [0-2]\n--energy [0-2]\n--pulse [0-2]",
        inline=False
    )

    await ctx.send(embed=embed)

def hex_to_rgba(hex_color):
    """Convert hex color to RGBA tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)) + (255,)

def main():
    if not DISCORD_TOKEN:
        print("Error: DISCORD_TOKEN not found in .env file")
        return
    print("Starting Image Generation bot...")
    bot.run(DISCORD_TOKEN)

if __name__ == "__main__":
    main()