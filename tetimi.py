import discord
from discord.ext import commands
from PIL import Image, ImageDraw, ImageOps, ImageEnhance, ImageFilter, ImageStat 
import numpy as np
from pathlib import Path
from io import BytesIO
from dotenv import load_dotenv
import os
import re
import random

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

# Enhanced effect presets with broader range
EFFECT_PRESETS = {
    'cyberpunk': {
        'rgb': (0, 255, 255),
        'alpha': 180,
        'chroma': 8,
        'glitch': 5,
        'scan': 2,
        'noise': 0.05
    },
    'vaporwave': {
        'rgb': (255, 100, 255),
        'alpha': 160,
        'chroma': 15,
        'scan': 3,
        'noise': 0.02
    },
    'glitch_art': {
        'rgb': (255, 50, 50),
        'alpha': 140,
        'glitch': 15,
        'chroma': 20,
        'noise': 0.1
    },
    'retro': {
        'rgb': (50, 200, 50),
        'alpha': 200,
        'scan': 2,
        'noise': 0.08
    },
    'matrix': {
        'rgb': (0, 255, 0),
        'alpha': 160,
        'scan': 1,
        'glitch': 3,
        'noise': 0.03,
        'chroma': 5
    },
    'synthwave': {
        'rgb': (255, 0, 255),
        'alpha': 180,
        'chroma': 10,
        'scan': 4,
        'noise': 0.02
    },
    'akira': {
        'rgb': (255, 0, 0),
        'alpha': 200,
        'chroma': 12,
        'glitch': 8,
        'scan': 1,
        'noise': 0.08
    },
    'tetsuo': {
        'rgb': (255, 50, 255),
        'alpha': 220,
        'chroma': 15,
        'glitch': 10,
        'scan': 2,
        'noise': 0.1
    }
}

# Set up Discord intents
intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True
bot = commands.Bot(command_prefix='!', intents=intents)

class ImageAnalyzer:
    @staticmethod
    def analyze_image(image):
        """Comprehensive image analysis for adaptive processing"""
        # Convert to RGB for analysis
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Calculate various image statistics
        stat = ImageStat.Stat(image)
        brightness = sum(stat.mean) / (3 * 255.0)
        contrast = sum(stat.stddev) / (3 * 255.0)
        
        # Calculate color dominance
        r, g, b = stat.mean
        color_variance = np.std([r, g, b])
        
        # Analyze image complexity
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
        """
        Generate adaptive parameters based on image analysis, but only for effects
        that are either requested by the user or included in the chosen preset.
        
        Parameters:
            analysis: Dictionary containing image analysis metrics
            user_params: Dictionary of parameters specified by user commands
            preset_params: Dictionary of parameters from selected preset
        
        Returns:
            Dictionary of adaptive parameters only for requested effects
        """
        params = {}
        
        # Create a set of requested effects from both user params and preset
        requested_effects = set()
        
        # Add effects from user parameters
        if user_params:
            requested_effects.update(user_params.keys())
        
        # Add effects from preset parameters
        if preset_params:
            requested_effects.update(preset_params.keys())
        
        # Only calculate parameters for requested effects
        if 'alpha' in requested_effects:
            # Adjust alpha based on image brightness and contrast
            params['alpha'] = int(255 * (1.0 - (analysis['brightness'] * 0.7 + 
                                               analysis['contrast'] * 0.3)))
        
        if 'glitch' in requested_effects:
            # Calculate glitch intensity based on image complexity
            params['glitch'] = int(20 * (1.0 - (analysis['complexity'] * 0.8)))
        
        if 'chroma' in requested_effects:
            # Determine chromatic aberration based on color variance
            params['chroma'] = int(20 * (1.0 - (analysis['color_variance']/255 * 0.9)))
        
        if 'noise' in requested_effects:
            # Set noise level based on contrast and brightness
            params['noise'] = float(analysis['contrast'] * 0.5 + 
                                  analysis['brightness'] * 0.5)
        
        if 'scan' in requested_effects:
            # Adjust scan line intensity based on image complexity
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
    
    # Apply slight blur for smoother transitions
    offset_image = offset_image.filter(ImageFilter.GaussianBlur(0.5))
    return offset_image

class ImageProcessor:
    def __init__(self, image_path):
        """Initialize with validation and analysis"""
        if not Path(image_path).exists():
            raise ValueError(f"Image file not found: {image_path}")
            
        if Path(image_path).stat().st_size > MAX_FILE_SIZE:
            raise ValueError(f"Image file too large (max {MAX_FILE_SIZE/1024/1024}MB)")
            
        self.base_image = Image.open(image_path).convert('RGBA')
        
        if not (MIN_IMAGE_SIZE[0] <= self.base_image.size[0] <= MAX_IMAGE_SIZE[0] and 
                MIN_IMAGE_SIZE[1] <= self.base_image.size[1] <= MAX_IMAGE_SIZE[1]):
            raise ValueError(f"Image dimensions must be between {MIN_IMAGE_SIZE} and {MAX_IMAGE_SIZE}")
            
        self.analyzer = ImageAnalyzer()
        self.analysis = self.analyzer.analyze_image(self.base_image)
        # Don't calculate adaptive params yet - wait for merge_params
        self.adaptive_params = {}
    
    def merge_params(self, user_params):
        """Merge user parameters with adaptive parameters"""
        # Get preset parameters if a preset was specified
        preset_name = user_params.get('preset')
        preset_params = EFFECT_PRESETS.get(preset_name, {})
        
        # Now get adaptive parameters only for requested effects
        self.adaptive_params = self.analyzer.get_adaptive_params(
            self.analysis,
            user_params,
            preset_params
        )
        
        # Merge the parameters with priority:
        # 1. User specified parameters
        # 2. Preset parameters
        # 3. Adaptive parameters
        merged = self.adaptive_params.copy()
        
        if preset_params:
            merged.update(preset_params)
        
        # User params take highest priority
        merged.update(user_params)
        
        return merged
    
    def add_color_overlay(self, color):
        """Add color overlay with enhanced blending"""
        if not isinstance(color, tuple) or len(color) != 4:
            raise ValueError("Invalid color format")
        if not all(isinstance(c, int) and 0 <= c <= 255 for c in color):
            raise ValueError("Color values must be integers between 0 and 255")
            
        colored = Image.new('RGBA', self.base_image.size, color)
        result = Image.new('RGBA', self.base_image.size, (0, 0, 0, 0))
        result.paste(colored, mask=self.mask)
        return result
    
    def apply_glitch_effect(self, intensity=10):
        """Enhanced glitch effect with smoother transitions"""
        if not isinstance(intensity, int) or not 1 <= intensity <= 20:
            raise ValueError("Glitch intensity must be an integer between 1 and 20")
            
        img_array = np.array(self.base_image)
        result = img_array.copy()
        
        for _ in range(intensity):
            offset = np.random.randint(-10, 10)
            if offset == 0:
                continue
                
            if offset > 0:
                result[offset:, :] = img_array[:-offset, :]
                result[:offset, :] = img_array[-offset:, :]
            else:
                offset = abs(offset)
                result[:-offset, :] = img_array[offset:, :]
                result[-offset:, :] = img_array[:offset, :]
            
            # Apply random channel shift
            channel = np.random.randint(0, 3)
            shift = np.random.randint(-5, 6)
            if shift != 0:
                temp = np.roll(result[:, :, channel], shift, axis=1)
                img_array[:, :, channel] = temp
        
        return Image.fromarray(result)
    

    def colorize_non_white(self, r, g, b, alpha=255):
        """
        Enhanced colorization with more nuanced color blending based on image brightness.
        The effect is strongest in darker areas and gradually reduces in brighter areas.
        
        Parameters:
            r, g, b: RGB color values (0-255) for the target color
            alpha: Overall intensity of the effect (0-255)
        """
        # Convert image to numpy array for faster processing
        img_array = np.array(self.base_image.convert('RGBA'))
        
        # Create our color overlay
        glow = img_array.copy()
        glow[:, :] = [r, g, b, alpha]
        
        # Calculate image brightness using perceptual color weights
        # These weights match human perception of color brightness
        luminance = np.sum(img_array[:, :, :3] * [0.299, 0.587, 0.114], axis=2)
        
        # Create mask for non-bright areas
        non_white_mask = luminance < 240
        
        # Create smooth transition based on brightness
        # The [:, :, np.newaxis] reshapes the array to work with RGB channels
        blend_factor = ((255 - luminance) / 255.0)[:, :, np.newaxis]
        
        # Intensify the effect while keeping it in valid range
        blend_factor = np.clip(blend_factor * 1.5, 0, 1)
        
        # Apply the color blend only to non-white areas
        result = img_array.copy()
        result[non_white_mask] = (
            (1 - blend_factor[non_white_mask]) * img_array[non_white_mask] +
            blend_factor[non_white_mask] * glow[non_white_mask]
        ).astype(np.uint8)
        
        return Image.fromarray(result)
    
    def add_chromatic_aberration(self, offset=10):
        """Enhanced chromatic aberration with improved color separation"""
        if not isinstance(offset, int) or not 1 <= offset <= 20:
            raise ValueError("Chromatic aberration offset must be between 1 and 20")
            
        r, g, b, a = self.base_image.split()
        
        # Apply progressive offsets
        r = offset_channel(r, -offset, 0)
        b = offset_channel(b, offset, 0)
        
        # Enhance green channel slightly
        g = offset_channel(g, int(offset * 0.3), 0)
        
        return Image.merge('RGBA', (r, g, b, a))
    
    def add_scan_lines(self, gap=2, alpha=128):
        """
        Enhanced scan lines with variable intensity and subtle glow effect.
        
        Parameters:
            gap: Space between scan lines (1-10)
            alpha: Base intensity of the lines (0-255)
        """
        width, height = self.base_image.size
        scan_lines = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(scan_lines)
        
        # Create scan lines with varying intensity
        for y in range(0, height, gap):
            # Calculate random intensity variation for each line
            intensity = int(alpha * (0.7 + 0.3 * random.random()))
            
            # Draw main scan line
            draw.line([(0, y), (width, y)], fill=(0, 0, 0, intensity))
            
            # Draw slightly fainter line above for glow effect
            if y > 0:  # Prevent drawing above image bounds
                draw.line([(0, y-1), (width, y-1)], 
                         fill=(0, 0, 0, intensity//2))
        
        # Apply subtle blur for glow effect
        scan_lines = scan_lines.filter(ImageFilter.GaussianBlur(0.5))
        
        # Composite the scan lines over the original image
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
    """Parse Discord command arguments with validation"""
    result = {}
    args = ' '.join(args)
    
    # Handle random flag
    if '--random' in args:
        images = list(Path(IMAGES_FOLDER).glob('*.*'))
        if not images:
            raise ValueError(f"No images found in {IMAGES_FOLDER}")
        result['image_path'] = str(random.choice(images))
    
    # Check for preset
    preset_match = re.search(r'--preset\s+(\w+)', args)
    if preset_match:
        preset_name = preset_match.group(1).lower()
        if preset_name in EFFECT_PRESETS:
            result.update(EFFECT_PRESETS[preset_name])
            return result
        raise ValueError(f"Unknown preset. Available presets: {', '.join(EFFECT_PRESETS.keys())}")
    
    # Parse RGB values and alpha
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
    
    # Parse other parameters
    params = {
        'glitch': (r'--glitch\s+(\d+)', lambda x: 1 <= x <= 20, "Glitch intensity must be between 1 and 20"),
        'chroma': (r'--chroma\s+(\d+)', lambda x: 1 <= x <= 20, "Chromatic aberration must be between 1 and 20"),
        'scan': (r'--scan\s+(\d+)', lambda x: 1 <= x <= 10, "Scan line gap must be between 1 and 10"),
        'noise': (r'--noise\s+(0?\.\d+)', lambda x: 0 <= float(x) <= 1, "Noise intensity must be between 0 and 1")
    }

    for param, (pattern, validator, error_msg) in params.items():
        match = re.search(pattern, args)
        if match:
            value = float(match.group(1)) if param == 'noise' else int(match.group(1))
            if validator(value):
                result[param] = value
            else:
                raise ValueError(error_msg)
    
    return result

    # Parse numeric parameters with validation
    params = {
        'glitch': (r'--glitch\s+(\d+)', lambda x: 1 <= x <= 20, "Glitch intensity must be between 1 and 20"),
        'chroma': (r'--chroma\s+(\d+)', lambda x: 1 <= x <= 20, "Chromatic aberration must be between 1 and 20"),
        'scan': (r'--scan\s+(\d+)', lambda x: 1 <= x <= 10, "Scan line gap must be between 1 and 10"),
        'noise': (r'--noise\s+(0?\.\d+)', lambda x: 0 <= float(x) <= 1, "Noise intensity must be between 0 and 1")
    }

    for param, (pattern, validator, error_msg) in params.items():
        match = re.search(pattern, args)
        if match:
            value = float(match.group(1)) if param == 'noise' else int(match.group(1))
            if validator(value):
                result[param] = value
            else:
                raise ValueError(error_msg)

    return result

@bot.event
async def on_ready():
    print(f'Tetsuo bot is online as {bot.user}')
    print(f"Using input image: {INPUT_IMAGE}")
    if not Path(INPUT_IMAGE).exists():
        print(f"Warning: Input image '{INPUT_IMAGE}' not found!")

@bot.event
async def on_reaction_add(reaction, user):
    if user != bot.user and str(reaction.emoji) == "🗑️":
        if reaction.message.author == bot.user:
            await reaction.message.delete()

@bot.command(name='tetsuo')
async def tetsuo_command(ctx, *args):
    try:
        params = parse_discord_args(args)
        image_path = params.pop('image_path', INPUT_IMAGE)
        
        if not Path(image_path).exists():
            await ctx.send(f"Error: Image '{image_path}' not found!")
            return
        
        params = parse_discord_args(args)
        if not params:
            presets_list = '\n'.join([f"- {name}: {', '.join(f'{k}={v}' for k, v in effects.items())}" 
                                    for name, effects in EFFECT_PRESETS.items()])
            await ctx.send(f"No valid arguments provided. Use effect parameters or try these presets:\n{presets_list}")
            return
        
        processor = ImageProcessor(image_path)
        
        # Always apply adaptive processing by merging with user params
        params = processor.merge_params(params)
        
        result = processor.base_image.convert('RGBA')
        
        effect_order = ['rgb', 'color', 'glitch', 'chroma', 'scan', 'noise']
        
        for effect in effect_order:
            if effect not in params:
                continue
            
            if effect == 'rgb':
                r, g, b = params['rgb']
                alpha = params.get('alpha', 255)
                result = processor.colorize_non_white(r, g, b, alpha)
            elif effect == 'color':
                color = hex_to_rgba(params['color'])
                color_result = processor.add_color_overlay(color)
                result = Image.alpha_composite(result, color_result)
            elif effect == 'glitch':
                processor.base_image = result.copy()
                glitch_result = processor.apply_glitch_effect(params['glitch'])
                if glitch_result.mode != 'RGBA':
                    glitch_result = glitch_result.convert('RGBA')
                result = glitch_result
            elif effect == 'chroma':
                processor.base_image = result.copy()
                result = processor.add_chromatic_aberration(params['chroma'])
            elif effect == 'scan':
                processor.base_image = result.copy()
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

def hex_to_rgba(hex_color):
    """Convert hex color to RGBA tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)) + (255,)

def main():
    if not DISCORD_TOKEN:
        print("Error: DISCORD_TOKEN not found in .env file")
        return
    print("Starting Tetsuo bot...")
    bot.run(DISCORD_TOKEN)

if __name__ == "__main__":
    main()