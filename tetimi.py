import discord
from discord.ext import commands
from PIL import Image, ImageDraw, ImageOps, ImageEnhance, ImageFilter
import numpy as np
from pathlib import Path
from io import BytesIO
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
INPUT_IMAGE = './images/tetsuo_logo.png'

# Constants for validation
MAX_IMAGE_SIZE = (2000, 2000)  # Maximum allowed image dimensions
MIN_IMAGE_SIZE = (50, 50)      # Minimum allowed image dimensions
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB max file size

# Effect presets
EFFECT_PRESETS = {
    'cyberpunk': {
        'rgb': (0, 255, 255),  # Cyan base
        'alpha': 180,          # Transparent enough to see detail
        'chroma': 8,
        'glitch': 5,
        'scan': 2,
        'noise': 0.05
    },
    'vaporwave': {
        'rgb': (255, 100, 255),  # Pink base
        'alpha': 160,            # More transparent for dreamy effect
        'chroma': 15,
        'scan': 3,
        'noise': 0.02
    },
    'glitch_art': {
        'rgb': (255, 50, 50),    # Red base
        'alpha': 140,            # Very transparent for glitch blend
        'glitch': 15,
        'chroma': 20,
        'noise': 0.1
    },
    'retro': {
        'rgb': (50, 200, 50),    # Green terminal
        'alpha': 200,            # Semi-transparent for CRT look
        'scan': 2,
        'noise': 0.08
    },
    'matrix': {
        'rgb': (0, 255, 0),      # Matrix green
        'alpha': 160,            # Transparent to maintain detail
        'scan': 1,
        'glitch': 3,
        'noise': 0.03,
        'chroma': 5              # Slight color separation
    },
    'synthwave': {
        'rgb': (255, 0, 255),    # Magenta
        'alpha': 180,            # Semi-transparent
        'chroma': 10,
        'scan': 4,
        'noise': 0.02
    }
}

# Set up Discord intents
intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True
bot = commands.Bot(command_prefix='!', intents=intents)

def offset_channel(image, offset_x, offset_y):
    """Custom function to offset an image channel"""
    width, height = image.size
    offset_image = Image.new(image.mode, (width, height), 0)
    
    # Calculate the actual offset positions with wrapping
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
    
    return offset_image

class ImageProcessor:
    def __init__(self, image_path):
        """Initialize with validation"""
        if not Path(image_path).exists():
            raise ValueError(f"Image file not found: {image_path}")
            
        if Path(image_path).stat().st_size > MAX_FILE_SIZE:
            raise ValueError(f"Image file too large (max {MAX_FILE_SIZE/1024/1024}MB)")
            
        self.base_image = Image.open(image_path).convert('RGBA')
        
        if not (MIN_IMAGE_SIZE[0] <= self.base_image.size[0] <= MAX_IMAGE_SIZE[0] and 
                MIN_IMAGE_SIZE[1] <= self.base_image.size[1] <= MAX_IMAGE_SIZE[1]):
            raise ValueError(f"Image dimensions must be between {MIN_IMAGE_SIZE} and {MAX_IMAGE_SIZE}")
            
        self.mask = self.base_image.convert('L')
        
    def add_color_overlay(self, color):
        """Add color overlay with validation"""
        if not isinstance(color, tuple) or len(color) != 4:
            raise ValueError("Invalid color format")
        if not all(isinstance(c, int) and 0 <= c <= 255 for c in color):
            raise ValueError("Color values must be integers between 0 and 255")
            
        colored = Image.new('RGBA', self.base_image.size, color)
        result = Image.new('RGBA', self.base_image.size, (0, 0, 0, 0))
        result.paste(colored, mask=self.mask)
        return result
    
    def apply_glitch_effect(self, intensity=10):
        """Apply glitch effect with validation"""
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
            
            channel = np.random.randint(0, 3)
            img_array[:, :, channel] = result[:, :, channel]
        
        return Image.fromarray(img_array)
    
    def colorize_non_white(self, r, g, b, alpha=255):
        """Colorize non-white pixels with alpha support"""
        if not all(isinstance(x, int) and 0 <= x <= 255 for x in [r, g, b, alpha]):
            raise ValueError("RGB and alpha values must be integers between 0 and 255")
            
        # Convert to RGBA array
        img_array = np.array(self.base_image.convert('RGBA'))
        
        # Create a color overlay with the specified alpha
        overlay = np.zeros_like(img_array)
        overlay[:, :] = [r, g, b, alpha]
        
        # Calculate luminance of original image
        luminance = np.sum(img_array[:, :, :3] * [0.299, 0.587, 0.114], axis=2)
        
        # Create mask for non-white pixels
        non_white_mask = luminance < 250  # Allow some tolerance for near-white pixels
        
        # Blend original with overlay based on luminance
        result = img_array.copy()
        blend_factor = alpha / 255.0
        
        # Only modify non-white pixels
        result[non_white_mask] = (
            (1 - blend_factor) * img_array[non_white_mask] +
            blend_factor * overlay[non_white_mask]
        ).astype(np.uint8)
        
        return Image.fromarray(result)
    
    def add_chromatic_aberration(self, offset=10):
        """Add RGB channel offset effect"""
        if not isinstance(offset, int) or not 1 <= offset <= 20:
            raise ValueError("Chromatic aberration offset must be between 1 and 20")
            
        r, g, b, a = self.base_image.split()
        r = offset_channel(r, -offset, 0)
        b = offset_channel(b, offset, 0)
        return Image.merge('RGBA', (r, g, b, a))
    
    def add_scan_lines(self, gap=2, alpha=128):
        """Add CRT-like scan lines"""
        if not isinstance(gap, int) or not 1 <= gap <= 10:
            raise ValueError("Scan line gap must be between 1 and 10")
        if not isinstance(alpha, int) or not 0 <= alpha <= 255:
            raise ValueError("Scan line alpha must be between 0 and 255")
            
        width, height = self.base_image.size
        scan_lines = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(scan_lines)
        
        for y in range(0, height, gap):
            draw.line([(0, y), (width, y)], fill=(0, 0, 0, alpha))
            
        return Image.alpha_composite(self.base_image.convert('RGBA'), scan_lines)
    
    def add_noise(self, intensity=0.1):
        """Add noise effect"""
        if not isinstance(intensity, (int, float)) or not 0 <= intensity <= 1:
            raise ValueError("Noise intensity must be between 0 and 1")
            
        img_array = np.array(self.base_image)
        noise = np.random.normal(0, intensity * 255, img_array.shape)
        noisy_image = img_array + noise
        np.clip(noisy_image, 0, 255, out=noisy_image)
        return Image.fromarray(noisy_image.astype('uint8'))

def parse_discord_args(args):
    """Parse Discord command arguments with validation"""
    result = {}
    args = ' '.join(args)
    
    # Check for preset first
    preset_match = re.search(r'--preset\s+(\w+)', args)
    if preset_match:
        preset_name = preset_match.group(1).lower()
        if preset_name in EFFECT_PRESETS:
            return EFFECT_PRESETS[preset_name]
        else:
            raise ValueError(f"Unknown preset. Available presets: {', '.join(EFFECT_PRESETS.keys())}")
    
    # Parse RGB values and alpha
    rgb_match = re.search(r'--rgb\s+(\d+)\s+(\d+)\s+(\d+)(?:\s+--alpha\s+(\d+))?', args)
    if rgb_match:
        r, g, b = map(int, rgb_match.groups()[:3])
        alpha = 255  # Default alpha
        if rgb_match.group(4):
            alpha = int(rgb_match.group(4))
            if not 0 <= alpha <= 255:
                raise ValueError("Alpha value must be between 0 and 255")
        if all(0 <= x <= 255 for x in [r, g, b]):
            result['rgb'] = (r, g, b)
            result['alpha'] = alpha
        else:
            raise ValueError("RGB values must be between 0 and 255")
    
    # Parse hex color
    color_match = re.search(r'--color\s+(#[0-9a-fA-F]{6})', args)
    if color_match:
        result['color'] = color_match.group(1)
    
    # Parse other effects
    for param, pattern, validator, error_msg in [
        ('glitch', r'--glitch\s+(\d+)', lambda x: 1 <= x <= 20, "Glitch intensity must be between 1 and 20"),
        ('chroma', r'--chroma\s+(\d+)', lambda x: 1 <= x <= 20, "Chromatic aberration offset must be between 1 and 20"),
        ('scan', r'--scan\s+(\d+)', lambda x: 1 <= x <= 10, "Scan line gap must be between 1 and 10"),
        ('noise', r'--noise\s+(0?\.\d+)', lambda x: 0 <= x <= 1, "Noise intensity must be between 0 and 1")
    ]:
        match = re.search(pattern, args)
        if match:
            value = float(match.group(1)) if param == 'noise' else int(match.group(1))
            if validator(value):
                result[param] = value
            else:
                raise ValueError(error_msg)
    
    return result

def hex_to_rgba(hex_color):
    """Convert hex color to RGBA tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)) + (255,)

@bot.event
async def on_ready():
    print(f'Tetsuo bot is online as {bot.user}')
    if not Path(INPUT_IMAGE).exists():
        print(f"Warning: Input image '{INPUT_IMAGE}' not found!")

@bot.event
async def on_reaction_add(reaction, user):
    if user != bot.user and str(reaction.emoji) == "🗑️":
        if reaction.message.author == bot.user:
            await reaction.message.delete()

@bot.command(name='tetsuo')
async def tetsuo_command(ctx, *args):
    """
    Apply image effects with optional alpha transparency
    
    --rgb R G B        Custom color values in decimal
    --alpha 0..255     Transparency for RGB effect (default: 255)
    --chroma 1..20     Chromatic prism effect
    --scan 1..20       Scan line distance
    --glitch 1..20     Split and offset color layers
    --noise 0.0..1.0   Noise percentage
    """
    if not Path(INPUT_IMAGE).exists():
        await ctx.send(f"Error: Input image '{INPUT_IMAGE}' not found!")
        return
    
    try:
        params = parse_discord_args(args)
        if not params:
            presets_list = '\n'.join([f"- {name}: {', '.join(f'{k}={v}' for k, v in effects.items())}" 
                                    for name, effects in EFFECT_PRESETS.items()])
            await ctx.send(f"No valid arguments provided. Use effect parameters or try these presets:\n{presets_list}")
            return
            
        processor = ImageProcessor(INPUT_IMAGE)
        result = processor.base_image.convert('RGBA')
        
        # Define effect order for consistent application
        effect_order = ['rgb', 'color', 'glitch', 'chroma', 'scan', 'noise']
        
        # Apply effects in consistent order
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
                temp_processor = ImageProcessor(INPUT_IMAGE)
                temp_processor.base_image = result.copy()
                glitch_result = temp_processor.apply_glitch_effect(params['glitch'])
                if glitch_result.mode != 'RGBA':
                    glitch_result = glitch_result.convert('RGBA')
                result = glitch_result
            elif effect == 'chroma':
                temp_processor = ImageProcessor(INPUT_IMAGE)
                temp_processor.base_image = result.copy()
                result = temp_processor.add_chromatic_aberration(params['chroma'])
            elif effect == 'scan':
                temp_processor = ImageProcessor(INPUT_IMAGE)
                temp_processor.base_image = result.copy()
                result = temp_processor.add_scan_lines(params['scan'])
            elif effect == 'noise':
                temp_processor = ImageProcessor(INPUT_IMAGE)
                temp_processor.base_image = result.copy()
                noise_result = temp_processor.add_noise(params['noise'])
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

@bot.command(name='tetsuo_presets')
async def list_presets(ctx):
    """Command to list all available presets and their effects"""
    response = "Available presets:\n\n"
    for name, effects in EFFECT_PRESETS.items():
        response += f"**{name}**\n"
        for effect, value in effects.items():
            response += f"- {effect}: {value}\n"
        response += "\n"
    await ctx.send(response)

def main():
    if not DISCORD_TOKEN:
        print("Error: DISCORD_TOKEN not found in .env file")
        return
    print("Starting Tetsuo bot...")
    print(f"Using input image: {INPUT_IMAGE}")
    bot.run(DISCORD_TOKEN)

if __name__ == "__main__":
    main()
