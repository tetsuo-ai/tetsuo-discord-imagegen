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

# Load environment variables
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
IMAGES_FOLDER = 'images'
INPUT_IMAGE = 'input.png'

# Create images folder if it doesn't exist
Path(IMAGES_FOLDER).mkdir(exist_ok=True)

# Constants for validation
MAX_IMAGE_SIZE = (2000, 2000)
MIN_IMAGE_SIZE = (50, 50)
MAX_FILE_SIZE = 8 * 1024 * 1024

# Define effect order
EFFECT_ORDER = ['rgb', 'color', 'glitch', 'chroma', 'scan', 'noise', 'energy', 'pulse']

# Enhanced effect presets with broader range
EFFECT_PRESETS = {
    'cyberpunk': {
        'rgb': (20, 235, 215),
        'alpha': 140,
        'chroma': 5,
        'glitch': 3,
        'scan': 80,
        'noise': 0.03
    },
    'vaporwave': {
        'rgb': (235, 100, 235),
        'alpha': 120,
        'chroma': 8,
        'scan': 100,
        'noise': 0.02
    },
    'glitch_art': {
        'rgb': (235, 45, 75),
        'alpha': 140,
        'glitch': 12,
        'chroma': 15,
        'scan': 120,
        'noise': 0.08
    },
    'retro': {
        'rgb': (65, 215, 95),
        'alpha': 160,
        'scan': 90,
        'noise': 0.03
    },
    'matrix': {
        'rgb': (25, 225, 95),
        'alpha': 130,
        'scan': 70,
        'glitch': 2,
        'noise': 0.02,
        'chroma': 3
    },
    'synthwave': {
        'rgb': (225, 45, 235),
        'alpha': 150,
        'chroma': 7,
        'scan': 150,
        'noise': 0.02
    },
    'akira': {
        'rgb': (235, 25, 65),
        'alpha': 160,
        'chroma': 8,
        'glitch': 6,
        'scan': 180,
        'noise': 0.03
    },
    'tetsuo': {
        'rgb': (235, 45, 225),
        'alpha': 160,
        'chroma': 10,
        'glitch': 6,
        'scan': 160,
        'noise': 0.05,
        'pulse': 0.15
    },
    'neo_tokyo': {
        'rgb': (235, 35, 85),
        'alpha': 150,
        'chroma': 12,
        'glitch': 8,
        'scan': 140,
        'noise': 0.04,
        'pulse': 0.1
    },
    'psychic': {
        'rgb': (185, 25, 235),
        'alpha': 170,
        'chroma': 15,
        'glitch': 5,
        'scan': 130,
        'noise': 0.03,
        'energy': 0.2
    },
    'tetsuo_rage': {
        'rgb': (225, 24, 42),
        'alpha': 200,
        'chroma': 20,
        'glitch': 25,
        'scan': 50,
        'noise': 0.15,
        'energy': 0.2,
        'pulse': 0.2
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
        self.base_image = Image.open(image_path).convert('RGBA')
        """Initialize processor with optional dual image mode"""
        if silkscreen:
            self.base_image = self.apply_silkscreen_effect()
        else:
            if not Path(image_path).exists():
                raise ValueError(f"Image file not found: {image_path}")
                
            if Path(image_path).stat().st_size > MAX_FILE_SIZE:
                raise ValueError(f"Image file too large (max {MAX_FILE_SIZE/1024/1024}MB)")
                
        # Size validation applies to both modes
        if not (MIN_IMAGE_SIZE[0] <= self.base_image.size[0] <= MAX_IMAGE_SIZE[0] and 
                MIN_IMAGE_SIZE[1] <= self.base_image.size[1] <= MAX_IMAGE_SIZE[1]):
            raise ValueError(f"Image dimensions must be between {MIN_IMAGE_SIZE} and {MAX_IMAGE_SIZE}")
            
        self.analyzer = ImageAnalyzer()
        self.analysis = self.analyzer.analyze_image(self.base_image)
        self.adaptive_params = {}

    def apply_silkscreen_effect(self, colors=None, dot_size=5, registration_offset=2):
        """
        Creates a denser Warhol-style silkscreen effect while maintaining dot clarity.
        
        Args:
            colors: List of hex colors for the separations. If None, uses defaults
            dot_size: Base size of halftone dots in pixels
            registration_offset: Simulated misregistration between color layers
        """
        if colors is None:
            # Maintain muted colors but increase initial density
            colors = ['#E62020', '#20B020', '#2020E6', '#D4D420']
        
        # Increase contrast slightly for better dot density in mid-tones
        image = self.base_image.convert('L')
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)  # Increased from 1.8
        
        # Start with white background
        result = Image.new('RGBA', image.size, (255, 255, 255, 255))
        
        for i, color in enumerate(colors):
            r, g, b = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            halftone = Image.new('RGBA', image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(halftone)
            
            # Adjust density factor for better coverage
            density_factor = 0.8 if color.startswith('#D4') else 1.0  # Increased yellow density
            
            for y in range(0, image.size[1], dot_size):
                for x in range(0, image.size[0], dot_size):
                    box = (x, y, min(x + dot_size, image.size[0]), 
                          min(y + dot_size, image.size[1]))
                    region = image.crop(box)
                    average = ImageStat.Stat(region).mean[0]
                    
                    # Enhanced brightness calculation for better mid-tone coverage
                    brightness_factor = ((255 - average) / 255.0) ** 0.8  # Added power factor for mid-tone boost
                    adjusted_brightness = brightness_factor * density_factor
                    
                    # Increased dot size factor from 0.4 to 0.6
                    dot_radius = int(adjusted_brightness * dot_size * 0.4)
                    
                    if dot_radius > 0:
                        # Reduced random offset for tighter pattern
                        offset_x = random.randint(-1, 1) * 0.5
                        offset_y = random.randint(-1, 1) * 0.5
                        center = (x + dot_size//2 + offset_x, 
                                y + dot_size//2 + offset_y)
                        
                        # Increased base opacity
                        opacity = int(255 * min(adjusted_brightness * 0.9, 1.0))
                        draw.ellipse([center[0] - dot_radius, 
                                    center[1] - dot_radius,
                                    center[0] + dot_radius, 
                                    center[1] + dot_radius], 
                                    fill=(r, g, b, opacity))
            
            # Apply registration offset
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
        """Add color overlay with enhanced blending"""
        if not isinstance(color, tuple) or len(color) != 4:
            raise ValueError("Invalid color format")
        if not all(isinstance(c, int) and 0 <= c <= 255 for c in color):
            raise ValueError("Color values must be integers between 0 and 255")
            
        colored = Image.new('RGBA', self.base_image.size, color)
        mask = self.base_image.convert('L')
        result = Image.new('RGBA', self.base_image.size, (0, 0, 0, 0))
        result.paste(colored, mask=mask)
        return result
        
    def apply_energy_effect(self, intensity=0.8):
        """Refined energy effect with better visual balance"""
        if not 0 <= intensity <= 2:
            raise ValueError("Energy intensity must be between 0 and 2")
            
        base = self.base_image.convert('RGBA')
        width, height = base.size
        
        # Reduced number of lines and increased minimum spacing
        num_lines = int(50 * intensity)  # Reduced from 100
        min_spacing = max(10, int(20 * (1 - intensity)))
        
        energy = Image.new('RGBA', base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(energy)
        
        prev_points = []
        for _ in range(num_lines):
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            
            # Ensure minimum spacing from previous lines
            if prev_points and any(abs(x1 - px) + abs(y1 - py) < min_spacing 
                                 for px, py in prev_points):
                continue
                
            angle = random.uniform(0, 2 * math.pi)
            length = random.randint(50, 150)
            x2 = x1 + int(length * math.cos(angle))
            y2 = y1 + int(length * math.sin(angle))
            
            # More controlled color selection
            hue = random.uniform(0.5, 0.7)  # More focused color range
            saturation = random.uniform(0.8, 1.0)
            value = random.uniform(0.8, 1.0)
            rgb = tuple(int(x * 255) for x in colorsys.hsv_to_rgb(hue, saturation, value))
            
            alpha = min(200, int(150 * intensity))
            color = rgb + (alpha,)
            
            line_width = max(1, int(2 * intensity))
            draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width)
            
            prev_points.append((x1, y1))
            if len(prev_points) > 5:  # Keep track of last 5 points only
                prev_points.pop(0)
        
        # Refined blur based on intensity
        blur_radius = min(3, 1 + intensity)
        energy = energy.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Blend more subtly with base image
        result = Image.blend(base, 
                           Image.alpha_composite(base, energy), 
                           min(0.7, intensity * 0.5))
        
        return result

    def apply_pulse_effect(self, intensity=0.7):
        """Refined pulse effect with better visual integration"""
        if not 0 <= intensity <= 2:
            raise ValueError("Pulse intensity must be between 0 and 2")
            
        base = self.base_image.convert('RGBA')
        pulse = Image.new('RGBA', base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(pulse)
        
        width, height = base.size
        num_spots = int(10 * intensity)  # Reduced from 20
        
        # Calculate image brightness to place pulses in darker areas
        brightness = ImageStat.Stat(base.convert('L')).mean[0]
        
        for _ in range(num_spots):
            x = random.randint(0, width)
            y = random.randint(0, height)
            
            # Adjust radius based on image brightness
            base_radius = int(30 * (1 - brightness/255))
            radius = random.randint(base_radius, base_radius * 2)
            
            # Create graduated pulse
            for r in range(radius, 0, -2):
                alpha = int(100 * (r/radius) * intensity * (1 - brightness/255))
                draw.ellipse([x-r, y-r, x+r, y+r], 
                           fill=(255, 255, 255, alpha))
        
        # Refined blur and blend
        pulse = pulse.filter(ImageFilter.GaussianBlur(radius=3))
        result = Image.blend(base, 
                           Image.alpha_composite(base, pulse), 
                           min(0.5, intensity * 0.3))
        
        return result

    def generate_ascii_art(self, target_width=100):
        """
        Generates ASCII art with proper character spacing and proportions,
        compensating for monospace font spacing characteristics.

        Args:
            target_width (int): Desired width in characters

        Returns:
            PIL.Image.Image: RGB image containing the rendered ASCII art
        """
        # Define ASCII characters in increasing order of intensity
        ASCII_CHARS = ' .:-=+*#%@'

        # Convert to grayscale
        image = self.base_image.convert('L')

        # Get original dimensions and aspect ratio
        original_width, original_height = image.size
        original_aspect_ratio = original_height / original_width

        # Load default font and get character dimensions using getbbox()
        font = ImageFont.load_default()
        bbox = font.getbbox('A')  # Get bounding box for a single character
        char_width = bbox[2] - bbox[0]
        char_height = bbox[3] - bbox[1]

        # Correct aspect ratio with character proportions
        char_aspect_ratio = char_height / char_width
        corrected_aspect_ratio = original_aspect_ratio * char_aspect_ratio

        # Calculate target height based on corrected aspect ratio
        target_height = int(target_width * corrected_aspect_ratio)

        # Resize image to match target dimensions
        image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

        # Convert to numpy array for pixel processing
        pixels = np.array(image)

        # Generate ASCII string based on pixel intensity
        ascii_str = ''
        num_chars = len(ASCII_CHARS)
        for y in range(target_height):
            for x in range(target_width):
                # Map pixel intensity to an ASCII character
                intensity = pixels[y, x] / 255
                ascii_str += ASCII_CHARS[int(intensity * (num_chars - 1))]
            ascii_str += '\n'

        # Calculate output image dimensions
        padding = 10  # Padding for the rendered image
        output_width = int(target_width * char_width) + padding * 2
        output_height = int(target_height * char_height) + padding * 2

        # Create output image
        output = Image.new('RGB', (output_width, output_height), 'black')
        draw = ImageDraw.Draw(output)

        # Draw the ASCII art on the output image
        y = padding
        for line in ascii_str.split('\n'):
            if line.strip():  # Avoid empty lines
                draw.text((padding, y), line, fill='white', font=font)
            y += char_height

        return output


    
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
        """Enhanced chromatic aberration with support for higher offsets"""
        if not isinstance(offset, int) or not 1 <= offset <= 40:
            raise ValueError("Chromatic aberration offset must be between 1 and 40")
            
        r, g, b, a = self.base_image.split()
        
        r = offset_channel(r, int(-offset * 1.2), 0)
        b = offset_channel(b, int(offset * 1.2), 0)
        g = offset_channel(g, int(offset * 0.4), 0)
        
        return Image.merge('RGBA', (r, g, b, a))
    
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
    
    rgb_match = re.search(r'--rgb\s+(\d+)\s+(\d+)\s+(\d+)(?:\s+--alpha\s+(\d+))?', args)
    if rgb_match:
        r, g, b = map(int, rgb_match.groups()[:3])
        alpha = int(rgb_match.group(4)) if rgb_match.group(4) else 255
        if not 0 <= alpha <= 255:
            raise ValueError("Alpha value must be between 0 and 255")
        if all(0 <= x <= 255 for x in [r, g, b]):
            result['rgb'] = (r, g, b)
            result['alpha'] = alpha
        else:
            raise ValueError("RGB values must be between 0 and 255")
    
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
        silkscreen = params.pop('silkscreen', False)  # Extract silkscreen parameter
        image_path = params.pop('image_path', INPUT_IMAGE)
        
        # Create processor with dual mode if specified
        processor = ImageProcessor(image_path, silkscreen=silkscreen)
        
        if not Path(image_path).exists():
            await ctx.send(f"Error: Image '{image_path}' not found!")
            return
        
        if '--ascii' in args:
            result = processor.generate_ascii_art()
            output = BytesIO()
            result.save(output, format='PNG')
            output.seek(0)
            await ctx.send(file=discord.File(fp=output, filename='tetsuo_ascii.png'))
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
                alpha = params.get('alpha', 255)
                result = processor.colorize_non_white(r, g, b, alpha)
            elif effect == 'energy' and 'energy' in params:
                result = processor.apply_energy_effect(params['energy'])
            elif effect == 'pulse' and 'pulse' in params:
                result = processor.apply_pulse_effect(params['pulse'])
            elif effect == 'color':
                color = hex_to_rgba(params['color'])
                color_result = processor.add_color_overlay(color)
                result = Image.alpha_composite(result, color_result)
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
        message = await ctx.send(file=discord.File(fp=output, filename='tetsuo_output.png'))
        await message.add_reaction("🗑️")
        
    except Exception as e:
        await ctx.send(f"Error: {str(e)}")

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