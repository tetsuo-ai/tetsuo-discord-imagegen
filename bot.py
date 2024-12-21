import asyncio
import io
import os
import random
import re
import sys
import tempfile
import time
import traceback  # Add this import
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import discord
from discord.ext import commands
from dotenv import load_dotenv
from PIL import Image, ImageEnhance

from anims import AnimationProcessor, Keyframe
from artrepo import ArtRepository
from ascii_anim import ASCIIAnimationProcessor
from channelpass import ChannelPassAnimator
from tetimi import ANIMATION_PRESETS, EFFECT_ORDER, ImageProcessor

load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
IMAGES_FOLDER = 'images'
INPUT_IMAGE = 'input.png'

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# Set up Discord intents
intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True
bot = commands.Bot(command_prefix='!', intents=intents, help_command=None)

# Initialize the ArtRepository globally
art_repo = ArtRepository(db_path="art_repository.db", storage_path="art_storage")
    
''' Defined in anims.py
@dataclass
class Keyframe:
    time: float  # 0.0 to 1.0
    value: Any
    easing: str = 'linear'  # Default to linear interpolation
'''

def parse_discord_args(args, IMAGES_FOLDER: str = "images") -> dict:
    """Parse command arguments with enhanced animation and keyframe support"""
    result = {}
    args = ' '.join(args)
    
    # Preset handling - do this first so individual params can override preset values
    if '--preset' in args:
        preset_match = re.search(r'--preset\s+(\w+)', args)
        if preset_match:
            preset_name = preset_match.group(1).lower()
            if preset_name in ANIMATION_PRESETS:
                # Copy preset parameters into result
                result['preset'] = preset_name
                preset_params = ANIMATION_PRESETS[preset_name]['params']
                for effect, value in preset_params.items():
                    result[effect] = value
            else:
                available_presets = ', '.join(ANIMATION_PRESETS.keys()) if ANIMATION_PRESETS else "none"
                raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available_presets}")

    # Animation handling
    if '--animate' in args:
        result['animate'] = True
        if '--frames' in args:
            frames_match = re.search(r'--frames\s+(\d+)', args)
            if frames_match:
                result['frames'] = min(int(frames_match.group(1)), 120)
            else:
                result['frames'] = 30
        else:
            result['frames'] = 30
            
        if '--fps' in args:
            fps_match = re.search(r'--fps\s+(\d+)', args)
            if fps_match:
                result['fps'] = min(int(fps_match.group(1)), 60)
            else:
                result['fps'] = 24
        else:
            result['fps'] = 24

    # Handle alpha values
    alpha_params = {
        'alpha': re.search(r'--alpha\s+(\d+)', args),
        'coloralpha': re.search(r'--coloralpha\s+(\d+)', args),
        'rgbalpha': re.search(r'--rgbalpha\s+(\d+)', args)
    }
    
    for param_name, match in alpha_params.items():
        if match:
            value = int(match.group(1))
            if 0 <= value <= 255:
                result[param_name] = value
            else:
                raise ValueError(f"{param_name} value must be between 0 and 255")

    # Set default alpha if none provided
    if not any(param in result for param in alpha_params.keys()):
        result['alpha'] = 180
        
    # Points effect handling
    if '--points' in args:
        result['points'] = True
        dot_match = re.search(r'--dot-size\s+(\d+)', args)
        if dot_match:
            result['dot_size'] = int(dot_match.group(1))
        reg_match = re.search(r'--reg-offset\s+(\d+)', args)
        if reg_match:
            result['registration_offset'] = int(reg_match.group(1))
    
    # Random image handling
    if '--random' in args:
        images = list(Path(IMAGES_FOLDER).glob('*.*'))
        if not images:
            raise ValueError(f"No images found in {IMAGES_FOLDER}")
        result['image_path'] = str(random.choice(images))
        print("Image selected: ", result['image_path'])

    # Effect parameter handling with keyframe support
    params = {
        'glitch': (r'--glitch(?:\s+(\d*\.?\d+)|\s+\[([^\]]+)\])', lambda x: 0 <= float(x) <= 50,
            "Glitch intensity must be between 0 and 50"),
        'chroma': (r'--chroma(?:\s+(\d*\.?\d+)|\s+\[([^\]]+)\])', lambda x: 0 <= float(x) <= 40,
            "Chromatic aberration must be between 0 and 40"),
        'scan': (r'--scan(?:\s+(\d*\.?\d+)|\s+\[([^\]]+)\])', lambda x: 0 <= float(x) <= 200,
                "Scan line gap must be between 0 and 200"),
        'noise': (r'--noise(?:\s+(\d*\.?\d+)|\s+\[([^\]]+)\])', lambda x: 0 <= float(x) <= 1,
                 "Noise intensity must be between 0 and 1"),
        'energy': (r'--energy(?:\s+(\d*\.?\d+)|\s+\[([^\]]+)\])', lambda x: 0 <= float(x) <= 1,
                  "Energy intensity must be between 0 and 1"),
        'pulse': (r'--pulse(?:\s+(\d*\.?\d+)|\s+\[([^\]]+)\])', lambda x: 0 <= float(x) <= 1,
                 "Pulse intensity must be between 0 and 1"),
        'consciousness': (r'--consciousness(?:\s+(\d*\.?\d+)|\s+\[([^\]]+)\])', lambda x: 0 <= float(x) <= 1,
                        "Consciousness intensity must be between 0 and 1"),
        'impact': (r'--impact(?:\s+(\S+))', lambda x: 0 < len(str(x)) <= 48,
                        "IMPACT filter requires between 1 and 48 character of text"),
    }
    
    for param, (pattern, validator, error_msg) in params.items():
        match = re.search(pattern, args)
        if match:
            # Check if we have a single value or keyframe values
            if match.group(1):  # Single value
                try:
                    value = float(match.group(1))
                except ValueError:
                    value = str(match.group(1))

                if validator(value):
                    result[param] = value
                else:
                    raise ValueError(error_msg)
            elif match.group(2):  # Keyframe values
                try:
                    keyframe_values = [float(x.strip()) for x in match.group(2).split(',')]
                    # Convert to keyframes list
                    if len(keyframe_values) == 1:
                        # Single value treated as target for interpolation
                        result[param] = [
                            Keyframe(time=0.0, value=keyframe_values[0], easing='linear'),
                            Keyframe(time=1.0, value=keyframe_values[0], easing='linear')
                        ]
                    else:
                        # Multiple values create keyframes at equal intervals
                        keyframes = []
                        for i, value in enumerate(keyframe_values):
                            if not validator(value):
                                raise ValueError(error_msg)
                            time = i / (len(keyframe_values) - 1)
                            keyframes.append(Keyframe(time=time, value=value, easing='linear'))
                        result[param] = keyframes
                except ValueError:
                    raise ValueError(f"Invalid keyframe values for {param}")
    
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


async def create_animation(image_input: Union[bytes, Image.Image, BytesIO], 
            preset_name: str) -> Path | None:
    """Create animation from image using preset"""
    if preset_name not in ANIMATION_PRESETS:
        raise ValueError(f"Unknown animation preset: {preset_name}")
        
    preset = ANIMATION_PRESETS[preset_name]
    processor = AnimationProcessor(image_input)
    
    try:
        frames = processor.generate_frames(
            params=preset['params'],
            num_frames=preset.get('frames', 60)
        )
        
        video_path = processor.create_video(
            frame_rate=preset.get('fps', 30),
            output_name=f"{preset_name}.mp4"
        )
        
        return video_path
    finally:
        processor.cleanup()

@bot.command(name='animate')
async def animate(ctx, preset_name: str):
    """Create an animation using a preset"""
    if preset_name == "":
        # List available presets in a compact format
        presets_list = []
        for name, preset in ANIMATION_PRESETS.items():
            desc = preset.get('description', '').split('\n')[0]  # Get first line only
            presets_list.append(f"**{name}**: {desc[:100]}")  # Limit description length
            
        # Split into multiple messages if needed
        message = "Available animation presets:\n"
        current_chunk = []
        current_length = len(message)
        
        for preset_info in presets_list:
            if current_length + len(preset_info) + 2 > 1900:  # Leave room for formatting
                await ctx.send(message + "\n".join(current_chunk))
                current_chunk = []
                current_length = 0
                message = ""  # Reset for subsequent messages
            current_chunk.append(preset_info)
            current_length += len(preset_info) + 2  # +2 for newline
            
        if current_chunk:
            await ctx.send(message + "\n".join(current_chunk))
        return
        
    preset_name = preset_name.lower()
    if preset_name not in ANIMATION_PRESETS:
        await ctx.send(f"Unknown preset '{preset_name}'. Use !animate to see available presets.")
        return
        
    try:
        # Use attachment if provided, otherwise use input.png
        if ctx.message.attachments:
            attachment = ctx.message.attachments[0]
            image_input = await attachment.read()
        else:
            if not Path("input.png").exists():
                await ctx.send("No image provided and input.png not found!")
                return
            image_input = "input.png"
        
        # Convert preset parameters to keyframes
        preset = ANIMATION_PRESETS[preset_name]
        keyframe_params = {}
        
        for param, value in preset['params'].items():
            if isinstance(value, (list, tuple)) and len(value) == 2:
                # Create keyframes for start and end values
                keyframe_params[param] = [
                    Keyframe(time=0.0, value=value[0], easing='linear'),
                    Keyframe(time=1.0, value=value[1], easing='linear')
                ]
            else:
                # Static value - create same value for start and end
                keyframe_params[param] = [
                    Keyframe(time=0.0, value=value, easing='linear'),
                    Keyframe(time=1.0, value=value, easing='linear')
                ]
        
        # Send initial status in a shorter message
        status_msg = await ctx.send(f"Generating animation with {preset_name} preset...")
        
        # Initialize animation processor
        processor = AnimationProcessor(image_input)
        try:
            frames = processor.generate_frames(
                params=keyframe_params,
                num_frames=preset.get('frames', 30)
            )
            
            video_path = processor.create_video(
                frame_rate=preset.get('fps', 24),
                output_name=f"{preset_name}.mp4"
            )
            
            if video_path and video_path.exists():
                # Send video file first
                await ctx.send(file=discord.File(str(video_path)))
                
                # Then send parameter info in a separate message if needed
                param_info = []
                for param, value in preset['params'].items():
                    if isinstance(value, (list, tuple)):
                        param_info.append(f"{param}: {value[0]} → {value[1]}")
                    else:
                        param_info.append(f"{param}: {value}")
                        
                if param_info:
                    info_msg = f"Animation parameters:\n```\n{chr(10).join(param_info)}```"
                    if len(info_msg) <= 2000:
                        await ctx.send(info_msg)
                    else:
                        await ctx.send("Animation complete! (parameters omitted due to length)")
            else:
                await status_msg.edit(content="Failed to create animation")
                
        finally:
            processor.cleanup()
            
    except Exception as e:
        error_msg = str(e)
        if len(error_msg) > 1900:
            error_msg = error_msg[:1900] + "..."
        await ctx.send(f"Error creating animation: {error_msg}")



@bot.command(name='image')
async def image(ctx, *args):
    """Process image with effects and optional animation"""
    try:
        
        image_path = INPUT_IMAGE

        # Handle image input
        if ctx.message.attachments:
            image_input = await ctx.message.attachments[0].read()
        else:
            if '--random' in args:
                images = list(Path(IMAGES_FOLDER).glob('*.*'))
                if not images:
                    await ctx.send(f"Error: No images found in {IMAGES_FOLDER}")
                image_path = str(random.choice(images))
            if not Path("input.png").exists():
                await ctx.send("No image provided and input.png not found!")
                return
            image_input = image_path
        
        # Parse arguments
        params = parse_discord_args(args)
        
        # Initialize processor
        processor = ImageProcessor(image_input)
        
        # Handle animation if requested
        if params.get('animate'):
            status_msg = await ctx.send("Generating animation...")
            
            try:
                # If a preset is specified, use its parameters
                if 'preset' in params:
                    preset_name = params['preset'].lower()
                    if preset_name in ANIMATION_PRESETS:
                        preset = ANIMATION_PRESETS[preset_name]
                        animation_params = preset['params']
                    else:
                        await ctx.send(f"Unknown preset '{preset_name}'")
                        return
                else:
                    animation_params = params
                
                # Initialize animation processor
                anim_processor = AnimationProcessor(processor.base_image)
                
                try:
                    # Generate frames
                    frames = anim_processor.generate_frames(
                        params=animation_params,
                        num_frames=params.get('frames', 30)
                    )
                    
                    # Create video
                    video_path = anim_processor.create_video(
                        frame_rate=params.get('fps', 24),
                        output_name="animation.mp4"
                    )
                    
                    if video_path and video_path.exists():
                        await status_msg.edit(content="Animation complete!")
                        await ctx.send(file=discord.File(str(video_path)))
                    else:
                        await status_msg.edit(content="Failed to create animation")
                        
                finally:
                    anim_processor.cleanup()
                    
            except Exception as e:
                await status_msg.edit(content=f"Error creating animation: {str(e)}")
                return
                
        else:
            # Process static image
            result = processor.base_image.convert('RGBA')
            
            # Apply effects in order
            for effect in EFFECT_ORDER:
                if effect in params:
                    result = processor.apply_effect(effect, params)
                    processor.base_image = result
            
            # Send static image
            buffer = BytesIO()
            processor.base_image.save(buffer, format='PNG')
            buffer.seek(0)
            await ctx.send(file=discord.File(buffer, 'processed.png'))
            
    except Exception as e:
        await ctx.send(f"Error processing image: {str(e)}")
 
@bot.command(name='ascii')
async def ascii_art(ctx, *args):
    try:
        image_path = INPUT_IMAGE
        upscale = False
        
        # Process arguments
        if args.index("random") >= 0:
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
 
@bot.command(name='ascii_animate')
async def ascii_animate_command(ctx, *args):
    """Create ASCII art animation from image"""
    import subprocess  # Add at top of file with other imports
    
    try:
        # Parse arguments
        params = parse_discord_args(args)
        frames = params.pop('frames', 24)
        cols = params.pop('cols', 80)
        
        # Handle image input
        if ctx.message.attachments:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                await ctx.message.attachments[0].save(tmp.name)
                image_path = tmp.name
        else:
            image_path = INPUT_IMAGE

        if not Path(image_path).exists():
            await ctx.send("Error: Image not found!")
            return

        if '--random' in args:
            images = list(Path(IMAGES_FOLDER).glob('*.*'))
            if not images:
                raise ValueError(f"No images found in {IMAGES_FOLDER}")
            image_path = str(random.choice(images))
            print("Image selected: ", image_path)
        status_msg = await ctx.send("Generating ASCII animation...")
        progress_msg = await ctx.send("Frame 0/" + str(frames))
        
        try:
            processor = ASCIIAnimationProcessor(image_path)
            
            # Update progress periodically
            async def update_progress():
                while True:
                    frame_count = len(list(processor.frames_dir.glob("frame_*.png")))
                    await progress_msg.edit(content=f"Frame {frame_count}/{frames}")
                    if frame_count >= frames:
                        break
                    await asyncio.sleep(2)
            
            # Start progress updater
            progress_task = asyncio.create_task(update_progress())
            
            # Generate frames
            frame_paths = await processor.generate_frames(
                params=params,
                num_frames=frames,
                cols=cols
            )
            
            await progress_task
            await status_msg.edit(content="Creating video...")
            
            # Create video from frames
            output_path = processor.output_dir / "ascii_animation.mp4"
            
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-framerate', '24',
                '-i', str(processor.frames_dir / 'frame_%04d.png'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '23',
                str(output_path)
            ]
            
            # Run ffmpeg asynchronously
            process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.communicate()
            
            if output_path.exists():
                await status_msg.edit(content="ASCII animation complete!")
                await ctx.send(file=discord.File(str(output_path)))
                
                # Also send text version
                text_path = processor.output_dir / "ascii_animation.txt"
                if text_path.exists():
                    await ctx.send(file=discord.File(str(text_path)))
            else:
                await status_msg.edit(content="Failed to create animation video")
            
            await progress_msg.delete()
            
        finally:
            processor.cleanup()
            if ctx.message.attachments and 'tmp' in locals():
                os.unlink(tmp.name)
            
    except Exception as e:
        await ctx.send(f"Error: {str(e)}")

@bot.command(name='rgb_gif')
async def rgb_gif(ctx, *args):
    """Create RGB channel pass animation with keyframes
    Usage: !rgb_gif [--g value,value,value] [--b value,value,value] [--frames num] [--random] [--impact text]
    Default: Creates a looping animation where green moves left and blue moves right"""
    try:
        # Parse arguments with defaults matching original channelpass behavior
        params = {
            # Default: Green moves left (-0.5 -> 0 -> 0.5)
            'g_values': [-0.5, 0, 0.5],
            # Default: Blue moves right (0.5 -> 0 -> -0.5)
            'b_values': [0.5, 0, -0.5]
        }
        frames = 60
        use_random = False
        
        i = 0
        while i < len(args):
            arg = args[i].lower()
            if arg == '--g':
                if i + 1 < len(args):
                    params['g_values'] = [float(x) for x in args[i + 1].split(',')]
                    i += 2
                else:
                    await ctx.send("Missing values after --g")
                    return
            elif arg == '--b':
                if i + 1 < len(args):
                    params['b_values'] = [float(x) for x in args[i + 1].split(',')]
                    i += 2
                else:
                    await ctx.send("Missing values after --b")
                    return
            elif arg == '--frames':
                if i + 1 < len(args):
                    frames = min(int(args[i + 1]), 120)  # Cap at 120 frames
                    i += 2
                else:
                    await ctx.send("Missing value after --frames")
                    return
            elif arg == '--random':
                use_random = True
                i += 1
            elif arg == '--impact':
                if i + 1 < len(args):
                    if len(args[i+1]) > 48:
                        await ctx.send("IMPACT text must be less than 48 characters")
                        return
                    params['impact'] = args[i + 1]
                    i += 2
                else:
                    await ctx.send("Missing text for IMPACT")
                    return
            else:
                i += 1
        
        # Handle image input
        image_input = None
        
        if ctx.message.attachments:
            image_input = await ctx.message.attachments[0].read()
        elif use_random:
            images = list(Path(IMAGES_FOLDER).glob('*.*'))
            if not images:
                await ctx.send(f"No images found in {IMAGES_FOLDER}")
                return
            random_image = random.choice(images)
            image_input = str(random_image)
            await ctx.send(f"Using random image: {random_image.name}")
        else:
            if not Path("input.png").exists():
                await ctx.send("No image provided and input.png not found!")
                return
            image_input = "input.png"
        
        # Create status message
        status_msg = await ctx.send("Generating RGB channel pass animation...")
        
        # Initialize animator and generate frames
        animator = ChannelPassAnimator(image_input)
        try:
            gif_path = animator.generate_frames(params, num_frames=frames)
            
            if gif_path and gif_path.exists():
                # Check file size
                file_size = gif_path.stat().st_size
                if file_size > 8 * 1024 * 1024:  # 8MB Discord limit
                    await status_msg.edit(content="Warning: Generated GIF is large. Sending...")
                
                await ctx.send(file=discord.File(str(gif_path)))
                await status_msg.edit(content="Animation complete!")
                
                # Store in art repository if desired
                if not isinstance(image_input, str):
                    image_input = "uploaded image"
                artwork_id = art_repo.store_artwork(
                    str(gif_path),
                    title=f"RGB Channel Pass Animation ({Path(str(image_input)).stem})",
                    creator_id=str(ctx.author.id),
                    creator_name=str(ctx.author),
                    tags=['animation', 'rgb_gif'],
                    parameters=params
                )
                await ctx.send(f"Stored as artwork ID: {artwork_id}")
            else:
                await status_msg.edit(content="Failed to create animation")
        except Exception as e:
            await status_msg.edit(content=f"Error: {str(e)}")
        finally:
            animator.cleanup()
            
    except Exception as e:
        await ctx.send(f"Error: {str(e)}")

@bot.command(name='remix')
async def remix(ctx, reference_id: str, *args):
    """Remix an artwork from the repository with support for multiple animation types
    Usage: 
        Standard animation: !remix <id> --animate --preset name [--frames num] [--fps num]
        RGB animation: !remix <id> rgb_gif [--g value,value,value] [--b value,value,value] [--frames num]
    """
    try:
        # Get original artwork
        image, metadata = art_repo.get_artwork(reference_id)
        
        # Convert args to lowercase for consistent checking
        args_lower = [arg.lower() for arg in args]
        
        # Check if RGB animation is requested
        if 'rgb_gif' in args_lower:
            # Create status message
            status_msg = await ctx.send(f"Creating RGB animation remix of {reference_id}...")
            
            try:
                # Parse RGB animation parameters
                params = {
                    'g_values': [-0.5, 0, 0.5],  # Default: Green moves left
                    'b_values': [0.5, 0, -0.5]   # Default: Blue moves right
                }
                frames = 60
                
                i = 0
                while i < len(args):
                    arg = args[i].lower()
                    if arg == '--g':
                        if i + 1 < len(args):
                            params['g_values'] = [float(x) for x in args[i + 1].split(',')]
                            i += 2
                        else:
                            await ctx.send("Missing values after --g")
                            return
                    elif arg == '--b':
                        if i + 1 < len(args):
                            params['b_values'] = [float(x) for x in args[i + 1].split(',')]
                            i += 2
                        else:
                            await ctx.send("Missing values after --b")
                            return
                    elif arg == '--frames':
                        if i + 1 < len(args):
                            frames = min(int(args[i + 1]), 120)
                            i += 2
                        else:
                            await ctx.send("Missing value after --frames")
                            return
                    else:
                        i += 1
                
                # Initialize animator with the original image
                animator = ChannelPassAnimator(image)
                
                try:
                    # Generate animation
                    gif_path = animator.generate_frames(params, num_frames=frames)
                    
                    if gif_path and gif_path.exists():
                        # Check file size
                        file_size = gif_path.stat().st_size
                        if file_size > 8 * 1024 * 1024:  # 8MB Discord limit
                            await status_msg.edit(content="Warning: Generated GIF is large. Sending...")
                        
                        # Store the remixed animation
                        new_id = art_repo.store_artwork(
                            str(gif_path),
                            title=f"RGB Remix of {metadata['title']}",
                            creator_id=str(ctx.author.id),
                            creator_name=str(ctx.author),
                            parent_id=reference_id,
                            tags=['remix', 'animation', 'rgb_gif'] + metadata.get('tags', []),
                            parameters={
                                'rgb_gif': params,
                                'frames': frames,
                                'parent_params': metadata.get('parameters', {})
                            }
                        )
                        
                        await ctx.send(file=discord.File(str(gif_path)))
                        await status_msg.edit(content=f"Created RGB animation remix {new_id}")
                    else:
                        await status_msg.edit(content="Failed to create RGB animation remix")
                        
                finally:
                    animator.cleanup()
                    
            except Exception as e:
                await status_msg.edit(content=f"Error creating RGB animation: {str(e)}")
                traceback.print_exc()
                return
                
        # Check if standard animation is requested
        elif '--animate' in args_lower:
            try:
                # Parse arguments
                params = parse_discord_args(args)
            except ValueError as e:
                await ctx.send(str(e))
                return
                
            # Create status message
            status_msg = await ctx.send(f"Creating animated remix of {reference_id}...")
            
            try:
                # Initialize animation processor with the original image
                processor = AnimationProcessor(image)
                
                # Get animation parameters
                num_frames = params.get('frames', 30)
                frame_rate = params.get('fps', 24)
                
                # If a preset is specified, use its parameters
                if 'preset' in params:
                    preset_name = params['preset'].lower()
                    if preset_name in ANIMATION_PRESETS:
                        preset = ANIMATION_PRESETS[preset_name]
                        animation_params = preset['params']
                        num_frames = preset.get('frames', num_frames)
                        frame_rate = preset.get('fps', frame_rate)
                    else:
                        available_presets = ', '.join(ANIMATION_PRESETS.keys())
                        await ctx.send(f"Unknown preset '{preset_name}'. Available presets: {available_presets}")
                        return
                else:
                    animation_params = params
                
                # Generate frames
                frames = processor.generate_frames(
                    params=animation_params,
                    num_frames=num_frames
                )
                
                # Create video
                video_path = processor.create_video(
                    frame_rate=frame_rate,
                    output_name=f"remix_{reference_id}.mp4"
                )
                
                if video_path and video_path.exists():
                    # Store the remixed animation
                    new_id = art_repo.store_artwork(
                        str(video_path),
                        title=f"Animated Remix of {metadata['title']}",
                        creator_id=str(ctx.author.id),
                        creator_name=str(ctx.author),
                        parent_id=reference_id,
                        tags=['remix', 'animation'] + metadata.get('tags', []),
                        parameters={
                            'animation': animation_params,
                            'frames': num_frames,
                            'fps': frame_rate,
                            'parent_params': metadata.get('parameters', {})
                        }
                    )
                    
                    await ctx.send(file=discord.File(str(video_path)))
                    await status_msg.edit(content=f"Created animated remix {new_id}")
                else:
                    await status_msg.edit(content="Failed to create animation remix")
                    
            except Exception as e:
                await status_msg.edit(content=f"Error creating animation: {str(e)}")
                traceback.print_exc()
            finally:
                if 'processor' in locals():
                    processor.cleanup()
        
        else:
            # Handle static image remix
            try:
                params = parse_discord_args(args)
            except ValueError as e:
                await ctx.send(str(e))
                return
                
            processor = ImageProcessor(image)
            result = processor.base_image.convert('RGBA')
            
            # Apply effects in order
            for effect in EFFECT_ORDER:
                if effect in params:
                    result = processor.apply_effect(effect, params)
            
            # Store the remixed image
            output = BytesIO()
            result.save(output, format='PNG')
            output.seek(0)
            
            new_id = art_repo.store_artwork(
                output,
                title=f"Remix of {metadata['title']}",
                creator_id=str(ctx.author.id),
                creator_name=str(ctx.author),
                parent_id=reference_id,
                tags=['remix'] + metadata.get('tags', []),
                parameters={
                    'effects': params,
                    'parent_params': metadata.get('parameters', {})
                }
            )
            
            output.seek(0)
            await ctx.send(
                f"Created remix {new_id}",
                file=discord.File(fp=output, filename=f"remix_{new_id}.png")
            )
            
    except Exception as e:
        error_msg = f"Error creating remix: {str(e)}"
        await ctx.send(error_msg)
        traceback.print_exc()  # Print the full traceback for debugging
@bot.command(name='testanimate')
async def test_animate(ctx):
    """Validates the animation pipeline by testing all available presets"""
    try:
        status = await ctx.send("Starting animation system test...")
        
        for preset_name in ANIMATION_PRESETS.keys():
            try:
                await status.edit(content=f"Testing preset: {preset_name}")
                
                processor = AnimationProcessor(INPUT_IMAGE)
                frames = processor.generate_frames(ANIMATION_PRESETS[preset_name], num_frames=15)
                
                video_path = processor.create_video(
                    frame_rate=24,
                    output_name=f"test_{preset_name}.mp4"
                )
                
                if video_path and video_path.exists():
                    await ctx.send(
                        f"Preset {preset_name} test complete",
                        file=discord.File(str(video_path))
                    )
                else:
                    await ctx.send(f"Failed to create video for {preset_name}")
                    continue
                    
            except Exception as preset_error:
                await ctx.send(f"Error testing preset {preset_name}: {str(preset_error)}")
                continue
            finally:
                if 'processor' in locals():
                    processor.cleanup()
        
        await status.edit(content="Animation system test complete - all presets processed")
        
    except Exception as e:
        await ctx.send(f"Test failed: {str(e)}")
        return

@bot.command(name='store')
async def store_artwork(ctx, *, details: str = ""):
    """Store artwork with title, tags, and description
    Usage: !store "Title" #tag1 #tag2 Description here"""
    if not ctx.message.attachments:
        await ctx.send("Please attach an image!\n"
                      "Usage: !store \"Title\" #tag1 #tag2 Description here")
        return

    # Parse metadata from details
    metadata = {}
    
    # Look for title in quotes
    title_match = re.search(r'"([^"]+)"', details)
    if title_match:
        metadata['title'] = title_match.group(1)
    else:
        # Use filename as title if no title provided
        metadata['title'] = Path(ctx.message.attachments[0].filename).stem
        
    # Look for tags after #
    tags = re.findall(r'#(\w+)', details)
    if tags:
        metadata['tags'] = tags
    else:
        # Use title words as tags if no tags provided
        metadata['tags'] = metadata['title'].lower().split()
        
    # Everything else becomes description
    desc_text = re.sub(r'"[^"]+"', '', details)  # Remove title
    desc_text = re.sub(r'#\w+', '', desc_text)   # Remove tags
    desc_text = desc_text.strip()
    if desc_text:
        metadata['description'] = desc_text

    try:
        # Store the artwork
        artwork_id = art_repo.store_artwork(
            image=await ctx.message.attachments[0].read(),
            title=metadata['title'],
            creator_id=str(ctx.author.id),
            creator_name=ctx.author.name,
            tags=metadata['tags'],
            description=metadata.get('description', '')
        )

        # Create confirmation embed
        embed = discord.Embed(
            title="Artwork Stored Successfully",
            description=f"ID: {artwork_id}",
            color=discord.Color.green()
        )
        
        embed.add_field(
            name="Title",
            value=metadata['title'],
            inline=False
        )
        
        embed.add_field(
            name="Tags",
            value=", ".join(metadata['tags']) if metadata['tags'] else "None",
            inline=False
        )
        
        if 'description' in metadata:
            embed.add_field(
                name="Description",
                value=metadata['description'],
                inline=False
            )
            
        embed.set_footer(text=f"Created by {ctx.author.name}")

        # Send confirmation with thumbnail of stored image
        message = await ctx.send(embed=embed)
        
        # Add reaction for deletion
        await message.add_reaction("🗑️")

    except Exception as e:
        await ctx.send(f"Error storing artwork: {str(e)}")

@bot.command(name='process_store')
async def process_and_store(ctx, *args):
    """Process image with effects and store result"""
    try:
        # Parse arguments (effects to apply)
        params = parse_discord_args(args)
        
        # Check if an attachment is present
        if not ctx.message.attachments:
            await ctx.send("Please attach an image!")
            return
            
        # Get the attachment (image file)
        attachment = ctx.message.attachments[0]
        image_bytes = await attachment.read()
        base_image = Image.open(io.BytesIO(image_bytes))  # Open the image from bytes
        
        # Initialize the ImageProcessor with the base image
        processor = ImageProcessor(base_image)  # Create an instance of ImageProcessor with the image
        
        # Convert to RGBA (or you could skip this if not needed)
        result = processor.base_image.convert('RGBA')
        
        # Apply effects in order
        for effect in ['rgb', 'color', 'glitch', 'chroma', 'scan', 'noise', 'energy', 'pulse', 'impact']:
            if effect not in params:
                continue
            processor.base_image = result.copy()
            result = processor.apply_effect(effect, params)
            # Generate title and tags based on effects used
            effect_tags = [effect for effect in ['rgb', 'color', 'glitch', 'chroma', 'scan', 'noise', 'energy', 'pulse', "impact"] 
                          if effect in params]
            title = f"Processed_{time.strftime('%Y%m%d_%H%M%S')}"
            tags = ['processed'] + effect_tags
            
            # Store processed result in the repository
            artwork_id = art_repo.store_artwork(
                image=result,
                title=title,
                creator_id=str(ctx.author.id),
                creator_name=ctx.author.name,
                tags=tags,
                parameters=params
            )
            
            # Save the processed result and send it back to Discord
            output = io.BytesIO()
            result.save(output, format='PNG')
            output.seek(0)
            
            await ctx.send(
                f"Artwork processed and stored! ID: {artwork_id}",
                file=discord.File(fp=output, filename=f"{artwork_id}.png")
            )
            
    except Exception as e:
        await ctx.send(f"Error processing artwork: {str(e)}")

@bot.command(name='search')
async def search_artwork(ctx, *, query: str = ""):
    """Search for artwork by any metadata"""
    try:
        if not query:
            await ctx.send("Please provide a search term!\n"
                         "Usage: !search <term>\n"
                         "Searches across: titles, tags, descriptions, creators, and effects")
            return

        results = art_repo.search_artwork(query)
        
        if not results:
            await ctx.send("No artwork found matching those terms.")
            return
            
        # Create paginated embed for results
        embed = discord.Embed(
            title=f"Search Results for '{query}'",
            description=f"Found {len(results)} matches",
            color=discord.Color.blue()
        )
        
        for art in results[:10]:  # Show first 10 results in first page
            # Format timestamp
            timestamp = datetime.fromtimestamp(art['timestamp'])
            time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            
            # Format effects/parameters if present
            effects = []
            for effect, value in art['parameters'].items():
                if isinstance(value, tuple):
                    effects.append(f"{effect}: {value[0]}-{value[1]}")
                else:
                    effects.append(f"{effect}: {value}")
            
            field_value = (
                f"Creator: {art['creator_name']}\n"
                f"Created: {time_str}\n"
                f"Tags: {', '.join(art['tags'])}\n"
                f"Views: {art['views']}\n"
            )
            
            if effects:
                field_value += f"Effects: {', '.join(effects)}\n"
                
            if art['description']:
                field_value += f"Description: {art['description'][:100]}..."
                
            embed.add_field(
                name=f"{art['title']} (ID: {art['id']})",
                value=field_value,
                inline=False
            )
        
        if len(results) > 10:
            embed.set_footer(text=f"Showing 10 of {len(results)} results. Please refine your search for more specific results.")
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"Error searching artwork: {str(e)}")

@bot.command(name='update')
async def update_artwork(ctx, artwork_id: str, *, details: str = ""):
    """Update artwork metadata like title, tags, or description"""
    try:
        # First check if user is the creator
        _, metadata = art_repo.get_artwork(artwork_id)
        if str(ctx.author.id) != metadata['creator_id']:
            await ctx.send("You can only update your own artwork!")
            return

        # Parse update details
        updates = {}
        
        # Look for title in quotes
        title_match = re.search(r'"([^"]+)"', details)
        if title_match:
            updates['title'] = title_match.group(1)
            
        # Look for tags after #
        tags = re.findall(r'#(\w+)', details)
        if tags:
            updates['tags'] = tags
            
        # Everything else becomes description
        desc_text = re.sub(r'"[^"]+"', '', details)  # Remove title
        desc_text = re.sub(r'#\w+', '', desc_text)   # Remove tags
        desc_text = desc_text.strip()
        if desc_text:
            updates['description'] = desc_text

        if not updates:
            await ctx.send("No updates provided! Use:\n"
                         "- Title in quotes: \"New Title\"\n"
                         "- Tags with #: #tag1 #tag2\n"
                         "- Remaining text becomes description")
            return

        # Apply updates
        art_repo.update_artwork(artwork_id, **updates)
        
        # Confirm updates
        update_msg = "Updated artwork:\n"
        if 'title' in updates:
            update_msg += f"- Title: {updates['title']}\n"
        if 'tags' in updates:
            update_msg += f"- Tags: {', '.join(updates['tags'])}\n"
        if 'description' in updates:
            update_msg += f"- Description: {updates['description']}\n"
            
        await ctx.send(update_msg)

    except Exception as e:
        await ctx.send(f"Error updating artwork: {str(e)}")

@bot.command(name='history')
async def show_history(ctx, artwork_id: str):
    """Show modification history of artwork"""
    try:
        history = art_repo.get_artwork_history(artwork_id)
        original, metadata = art_repo.get_artwork(artwork_id)
        
        embed = discord.Embed(
            title=f"History for {metadata['title']}",
            description=f"Original creator: {metadata['creator_name']}",
            color=discord.Color.purple()
        )
        
        # Add original creation
        embed.add_field(
            name="Original Creation",
            value=f"Created by {metadata['creator_name']}\n"
                 f"Tags: {', '.join(metadata['tags'])}",
            inline=False
        )
        
        # Add modifications
        for entry in history:
            # Format timestamp
            timestamp = datetime.fromtimestamp(entry['timestamp'])
            time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            
            # Get modifier name
            modifier = await bot.fetch_user(int(entry['modifier']))
            
            # Format changes
            changes = []
            if 'title' in entry['changes']:
                changes.append(f"Title → {entry['changes']['title']}")
            if 'tags' in entry['changes']:
                changes.append(f"Tags → {', '.join(entry['changes']['tags'])}")
            if 'description' in entry['changes']:
                changes.append(f"Description updated")
            
            embed.add_field(
                name=f"Modified {time_str}",
                value=f"By: {modifier.name}\n"
                     f"Changes: {', '.join(changes)}",
                inline=False
            )
        
        await ctx.send(embed=embed)

    except Exception as e:
        await ctx.send(f"Error getting history: {str(e)}")


@bot.command(name='trending')
async def show_trending(ctx):
    """Show trending artwork"""
    try:
        trending = art_repo.get_trending_artwork()
        
        embed = discord.Embed(
            title="Trending Artwork",
            description="Most popular pieces in the last 24 hours",
            color=discord.Color.purple()
        )
        
        for artwork in trending:
            creator = await bot.fetch_user(int(artwork['creator']))
            embed.add_field(
                name=f"{artwork['title']} (ID: {artwork['id']})",
                value=f"Creator: {creator.name}\nInteractions: {artwork['total_interactions']}\n"
                      f"Unique Users: {artwork['unique_users']}",
                inline=False
            )
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"Error getting trending artwork: {str(e)}")

@bot.command(name='help')
async def help_command(ctx):
    embed = discord.Embed(
        title="Tetimi Image Bot Commands",
        description="Image effects generator inspired by Akira",
        color=discord.Color.purple()
    )
    
    embed.add_field(
        name="Basic Usage",
        value="!image [options] - Process default/attached image\n!image --random - Process random image",
        inline=False
    )
    
    embed.add_field(
        name="Presets",
        value="--preset [name]\nAvailable: cyberpunk, vaporwave, glitch_art, retro, matrix, synthwave, akira, tetsuo, neo_tokyo, psychic, tetsuo_rage",
        inline=False
    )
    
    embed.add_field(
        name="Effect Options",
        value=(
            "--rgb [r] [g] [b] --rgbalpha [0-255]\n"
            "--color [r] [g] [b] --coloralpha [0-255]\n"
            "--glitch [1-50]\n"
            "--chroma [1-40]\n"
            "--scan [1-200]\n"
            "--noise [0-1]\n"
            "--energy [0-1]\n"
            "--pulse [0-1]\n"
            "--consciousness [0-1]\n",
            "--impact [Text] (Max Len 48)"
        ),
        inline=False
    )
    
    embed.add_field(
        name="Special Effects",
        value=(
            "--points - Apply points effect\n"
            "--dot-size [size] - Points dot size\n"
            "--reg-offset [offset] - Points registration offset\n"
            "--animate - Create animation (with --frames [n] --fps [n])"
        ),
        inline=False
    )

    embed.add_field(
        name="Art Repository Commands",
        value=(
            "!store \"Title\" #tag1 #tag2 Description - Store artwork\n"
            "!search <query> - Search stored artwork\n"
            "!update <id> \"New Title\" #newtag - Update artwork\n"
            "!remix <id> [effects] - Remix existing artwork\n"
            "!history <id> - Show artwork history\n"
            "!trending - Show popular artwork"
        ),
        inline=False
    )

    await ctx.send(embed=embed)

@bot.event
async def on_shutdown():
    art_repo.conn.close()
    print("Database connection closed.")

def main():
    if not DISCORD_TOKEN:
        print("Error: DISCORD_TOKEN not found in .env file")
        print("Please create a .env file with your Discord bot token:")
        print("DISCORD_TOKEN=your_token_here")
        return
        
    # Create required directories
    os.makedirs(IMAGES_FOLDER, exist_ok=True)
    os.makedirs("art_storage", exist_ok=True)
    
    print("Starting Tetimi bot...")
    bot.run(DISCORD_TOKEN)

if __name__ == "__main__":
    main()
