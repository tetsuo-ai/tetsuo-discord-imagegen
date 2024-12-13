import discord
from discord.ext import commands
from PIL import Image, ImageFont, ImageDraw, ImageOps, ImageEnhance, ImageFilter, ImageStat 
import numpy as np
from pathlib import Path
from io import BytesIO
from dotenv import load_dotenv
import os
import re
import random
import colorsys

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
        'alpha': 180,
        'chroma': 8,
        'glitch': 5,
        'scan': 8,
        'noise': 0.05
        },
    'cyberpunk': {
        'rgb': (20, 235, 215),  # Cyan with slight adjustment
        'alpha': 180,
        'chroma': 8,
        'glitch': 5,
        'scan': 8,
        'noise': 0.05
    },
    'vaporwave': {
        'rgb': (235, 100, 235),  # Softer pink
        'alpha': 160,
        'chroma': 15,
        'scan': 8,
        'noise': 0.02
    },
    'glitch_art': {
        'rgb': (235, 45, 75),  # Warmer red
        'alpha': 140,
        'glitch': 15,
        'chroma': 20,
        'noise': 0.1
    },
    'retro': {
        'rgb': (65, 215, 95),  # Richer green
        'alpha': 200,
        'scan': 8,
        'noise': 0.05
    },
    'matrix': {
        'rgb': (25, 225, 95),  # More muted matrix green
        'alpha': 160,
        'scan': 6,
        'glitch': 3,
        'noise': 0.03,
        'chroma': 5
    },
    'synthwave': {
        'rgb': (225, 45, 235),  # Deeper magenta
        'alpha': 180,
        'chroma': 10,
        'scan': 12,
        'noise': 0.02
    },
    'akira': {
        'rgb': (235, 25, 65),  # Richer red
        'alpha': 200,
        'chroma': 12,
        'glitch': 8,
        'scan': 4,
        'noise': 0.05
    },
    'tetsuo': {
        'rgb': (235, 45, 225),  # Deeper purple-pink
        'alpha': 220,
        'chroma': 15,
        'glitch': 10,
        'scan': 8,
        'noise': 0.1,
    },
    'neo_tokyo': {
        'rgb': (235, 35, 85),  # Richer neon red
        'alpha': 190,
        'chroma': 18,
        'glitch': 12,
        'scan': 8,
        'noise': 0.08,
        'pulse': 0.2
    },
    'psychic': {
        'rgb': (185, 25, 235),  # Deeper purple
        'alpha': 170,
        'chroma': 25,
        'glitch': 8,
        'scan': 4,
        'noise': 0.05,
        'energy': 0.3
    },
    'tetsuo_rage': {
        'rgb': (225, 24, 42),
        'alpha': 255,
        'chroma': 40,
        'glitch': 45,
        'scan': 4,
        'noise': 0.3,
        'energy': 0.3,
        'pulse': 0.3
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
    def __init__(self, image_path):
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
        self.adaptive_params = {}
    
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
        """Enhanced energy effect with support for higher intensities"""
        if not 0 <= intensity <= 2:
            raise ValueError("Energy intensity must be between 0 and 2")
            
        base = self.base_image.convert('RGBA')
        width, height = base.size
        
        num_lines = int(100 * intensity)
        line_width = max(2, int(intensity * 3))
        
        energy = Image.new('RGBA', base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(energy)
        
        for _ in range(num_lines):
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = x1 + random.randint(-200, 200)
            y2 = y1 + random.randint(-200, 200)
            
            hue = random.uniform(0.8, 1.0)
            rgb = tuple(int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1, 1))
            color = rgb + (min(255, int(200 * intensity)),)
            
            draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width)
        
        blur_radius = min(5, 3 + intensity)
        energy = energy.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        return Image.alpha_composite(base, energy)

    def apply_pulse_effect(self, intensity=0.7):
        """Creates a pulsing light effect"""
        if not 0 <= intensity <= 2:
            raise ValueError("Pulse intensity must be between 0 and 2")
            
        pulse = Image.new('RGBA', self.base_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(pulse)
        
        width, height = self.base_image.size
        num_spots = int(20 * intensity)
        
        for _ in range(num_spots):
            x = random.randint(0, width)
            y = random.randint(0, height)
            radius = random.randint(20, 50)
            
            for r in range(radius, 0, -1):
                alpha = int(255 * (r/radius) * intensity)
                draw.ellipse([x-r, y-r, x+r, y+r], 
                           fill=(255, 255, 255, alpha))
        
        pulse = pulse.filter(ImageFilter.GaussianBlur(radius=5))
        return Image.alpha_composite(self.base_image, pulse)

    '''def generate_ascii_art(self, size=50):
        ASCII_CHARS = ' .:-=+*#%@'
        
        image = self.base_image.convert('L')
        
        aspect_ratio = image.size[0] / image.size[1]
        new_width = int(size * 2 * aspect_ratio)
        image = image.resize((new_width, size), Image.Resampling.LANCZOS)
        
        pixels = image.getdata()
        ascii_str = ''
        for i, pixel in enumerate(pixels):
            if i % new_width == 0:
                ascii_str += '\n'
            index = int((pixel / 255) * (len(ASCII_CHARS) - 1))
            ascii_str += ASCII_CHARS[index]
        
        font = ImageFont.load_default()
        bbox = font.getbbox('A')
        char_width = bbox[2] - bbox[0]
        char_height = bbox[3] - bbox[1]
        
        img_width = new_width * char_width
        img_height = size * char_height
        
        image = Image.new('RGB', (img_width, img_height), color='black')
        draw = ImageDraw.Draw(image)
        
        y = 0
        for line in ascii_str.split('\n'):
            draw.text((0, y), line, font=font, fill='white')
            y += char_height
        
        return image'''
   
    
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
    noise_match = re.search(r'--noise\s+(0?\.\d+)(?:\s+--style\s+(film|digital))?', args)
    
    if noise_match:
        intensity = float(noise_match.group(1))
        style = noise_match.group(2) or 'digital'  # Default to digital if style not specified
        
        if 0 <= intensity <= 1:
            result['noise'] = intensity
            result['noise_style'] = style
        else:
            raise ValueError("Noise intensity must be between 0 and 1")
            
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
        'glitch': (r'--glitch\s+(\d+)', lambda x: 1 <= x <= 50, "Glitch intensity must be between 1 and 50"),
        'chroma': (r'--chroma\s+(\d+)', lambda x: 1 <= x <= 40, "Chromatic aberration must be between 1 and 40"),
        'scan': (r'--scan\s+(\d+)', lambda x: 1 <= x <= 20, "Scan line gap must be between 1 and 20"),
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
    print(f'Tetsuo bot is online as {bot.user}')
    print(f"Using input image: {INPUT_IMAGE}")
    if not Path(INPUT_IMAGE).exists():
        print(f"Warning: Input image '{INPUT_IMAGE}' not found!")

@bot.event
async def on_reaction_add(reaction, user):
    if user != bot.user and str(reaction.emoji) == "🗑️":
        if reaction.message.author == bot.user:
            await reaction.message.delete()

@bot.command(name='image')
async def tetsuo_command(ctx, *args):
    try:
        params = parse_discord_args(args)
        image_path = params.pop('image_path', INPUT_IMAGE)
        
        if not Path(image_path).exists():
            await ctx.send(f"Error: Image '{image_path}' not found!")
            return
        
        processor = ImageProcessor(image_path)
        
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
            await ctx.send(f"No valid arguments provided. Use !tetsuo_help for full options")
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
async def tetsuo_help(ctx):
    embed = discord.Embed(
        title="Tetsuo Bot Commands",
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
    print("Starting Tetsuo bot...")
    bot.run(DISCORD_TOKEN)

if __name__ == "__main__":
    main()