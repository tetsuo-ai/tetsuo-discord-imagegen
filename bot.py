import discord
from discord.ext import commands
from pathlib import Path
from io import BytesIO
import os
import re
import time
from dotenv import load_dotenv
import asyncio
import sys
import tempfile
import random
from PIL import Image, ImageEnhance
from datetime import datetime
from tetimi import ImageProcessor, EFFECT_PRESETS, EFFECT_ORDER
from artrepo import ArtRepository
from anims import AnimationProcessor 
import io
from artrepo import ArtRepository
from typing import Dict, Any
from ascii_anim import ASCIIAnimationProcessor



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
    
def parse_discord_args(args, IMAGES_FOLDER: str = "images", EFFECT_PRESETS: Dict = None) -> Dict[str, Any]:
    """Parse command arguments with enhanced animation and preset support
    
    Args:
        args: Command arguments
        IMAGES_FOLDER: Path to images folder
        EFFECT_PRESETS: Dictionary of effect presets
        
    Returns:
        Dictionary of parsed parameters
    """
    result = {}
    args = ' '.join(args)
    
    # Preset handling - do this first so individual params can override preset values
    if '--preset' in args:
        preset_match = re.search(r'--preset\s+(\w+)', args)
        if preset_match:
            preset_name = preset_match.group(1).lower()
            if EFFECT_PRESETS and preset_name in EFFECT_PRESETS:
                # Copy preset values to result
                result.update(EFFECT_PRESETS[preset_name])
            else:
                available_presets = ', '.join(EFFECT_PRESETS.keys()) if EFFECT_PRESETS else "none"
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

    alpha_match = re.search(r'--alpha\s+(\d+)', args)
    coloralpha_match = re.search(r'--coloralpha\s+(\d+)', args)
    rgbalpha_match = re.search(r'--rgbalpha\s+(\d+)', args)

    # Handle alpha
    if alpha_match:
        alpha = int(alpha_match.group(1))
        if 0 <= alpha <= 255:
            result['coloralpha'] = alpha
        else:
            raise ValueError("Alpha (coloralpha) value must be between 0 and 255")
    
    # Handle coloralpha
    elif coloralpha_match:
        coloralpha = int(coloralpha_match.group(1))
        if 0 <= coloralpha <= 255:
            result['coloralpha'] = coloralpha
        else:
            raise ValueError("Coloralpha value must be between 0 and 255")
    
    # Handle rgbalpha
    elif rgbalpha_match:
        rgbalpha = int(rgbalpha_match.group(1))
        if 0 <= rgbalpha <= 255:
            result['rgbalpha'] = rgbalpha
        else:
            raise ValueError("Rgbalpha value must be between 0 and 255")

    # If none are provided, set default alpha
    elif 'alpha' not in result and 'coloralpha' not in result and 'rgbalpha' not in result:
        result['alpha'] = 180  # Default alpha value
        
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
    
    # Color parameter handling - can override preset values
    color_match = re.search(r'--(rgb|color)\s+(\d+)\s+(\d+)\s+(\d+)(?:\s+--\1alpha\s+(\d+))?', args)
    if color_match:
        color_type = color_match.group(1)
        r, g, b = map(int, color_match.groups()[1:4])
        alpha = int(color_match.group(5)) if color_match.group(5) else 255
        
        if all(0 <= x <= 255 for x in [r, g, b, alpha]):
            result[color_type] = (r, g, b)
            result[f'{color_type}alpha'] = alpha
        else:
            raise ValueError("Color/Alpha values must be between 0 and 255")
    
    # Effect parameter handling - can override preset values
    params = {
        'glitch': (r'--glitch\s+(\d*\.?\d+)', lambda x: 0 <= float(x) <= 50, 
                  "Glitch intensity must be between 0 and 50"),
        'chroma': (r'--chroma\s+(\d*\.?\d+)', lambda x: 0 <= float(x) <= 40, 
                  "Chromatic aberration must be between 0 and 40"),
        'scan': (r'--scan\s+(\d*\.?\d+)', lambda x: 0 <= float(x) <= 200, 
                "Scan line gap must be between 0 and 200"),
        'noise': (r'--noise\s+(\d*\.?\d+)', lambda x: 0 <= float(x) <= 10, 
                 "Noise intensity must be between 0 and 1"),
        'energy': (r'--energy\s+(\d*\.?\d+)', lambda x: 0 <= float(x) <= 1, 
                  "Energy intensity must be between 0 and 1"),
        'pulse': (r'--pulse\s+(\d*\.?\d+)', lambda x: 0 <= float(x) <= 1, 
                 "Pulse intensity must be between 0 and 1"),
        'consciousness': (r'--consciousness\s+(\d*\.?\d+)', lambda x: 0 <= float(x) <= 1,
                        "Consciousness intensity must be between 0 and 1")
    }
    
    for param, (pattern, validator, error_msg) in params.items():
        match = re.search(pattern, args)
        if match:
            value = float(match.group(1))
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
    """Process image with optional animation"""
    try:
        image_path = INPUT_IMAGE
        params = parse_discord_args(args, IMAGES_FOLDER, EFFECT_PRESETS)
        animate = params.pop('animate', False)
        frames = params.pop('frames', 24)
        fps = params.pop('fps', 24)
        # Process arguments
        if 'random' in args:
            images = list(Path(IMAGES_FOLDER).glob('*.*'))
            if not images:
                await ctx.send(f"Error: No images found in {IMAGES_FOLDER}")
                return
            image_path = str(random.choice(images))
           
        if not Path(image_path).exists():
            await ctx.send(f"Error: Image '{image_path}' not found!")
            return
                
        processor = ImageProcessor(image_path)

        # Handle image input
        if ctx.message.attachments:
            # If attachment exists, use it
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                await ctx.message.attachments[0].save(tmp.name)
                image_path = tmp.name
        if '--random' in args:
            image_path = ""
            image_path = params['image_path'] 
            
        if not Path(image_path).exists():
            await ctx.send(f"Error: Image not found!")
            return 
            
        if animate:
            status_msg = await ctx.send("Generating animation...")
            try:
                processor = AnimationProcessor(image_path)
                processor.generate_frames(params=params, num_frames=frames)
                video_path = processor.create_video(frame_rate=fps)  
                if video_path and video_path.exists():
                    await status_msg.edit(content="Animation complete! Uploading...")
                    await ctx.send(file=discord.File(str(video_path)))
                else:
                    await status_msg.edit(content="Failed to generate animation")
            finally:
                # Ensure cleanup
                processor.cleanup()
                
        else:
            # Process single image
            processor = ImageProcessor(image_path)
            result = processor.base_image.convert('RGBA')
            # Apply effects in order
            for effect in EFFECT_ORDER:
                if effect in params:
                    processor.base_image = result
                    result = processor.apply_effect(effect, params)

            # Save and send result
            output = BytesIO()
            result.save(output, format='PNG')
            output.seek(0)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_tetimi.png"
            message = await ctx.send(file=discord.File(fp=output, filename=filename))
            await message.add_reaction("🗑️")

        # Cleanup temporary file if it was created
        if ctx.message.attachments and 'tmp' in locals():
            os.unlink(tmp.name)

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
 
@bot.command(name='ascii_animate')
async def ascii_animate_command(ctx, *args):
    """Create ASCII art animation from image"""
    try:
        # Parse arguments
        params = parse_discord_args(args)
        frames = params.pop('frames', 24)
        
        # Handle image input
        if ctx.message.attachments:
            # If attachment exists, use it
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                await ctx.message.attachments[0].save(tmp.name)
                image_path = tmp.name
        else:
            # Default to INPUT_IMAGE
            image_path = INPUT_IMAGE

        if not Path(image_path).exists():
            await ctx.send("Error: Image not found!")
            return

        status_msg = await ctx.send("Generating ASCII animation...")
        
        try:
            processor = ASCIIAnimationProcessor(image_path)
            
            # Print params to check if 'cols' is removed
            print(f"Remaining params: {params}")  # Should not include 'cols'
            ascii_frames = processor.generate_ascii_frames(params=params, num_frames=frames, cols=80)
            output_path = processor.save_frames(ascii_frames)
            
            await status_msg.edit(content="ASCII animation complete! Uploading...")
            await ctx.send(file=discord.File(str(output_path)))
            
        finally:
            # Ensure cleanup
            processor.cleanup()
            if ctx.message.attachments and 'tmp' in locals():
                os.unlink(tmp.name)
            
    except Exception as e:
        await ctx.send(f"Error: {str(e)}")


@bot.command(name='testanimate')
async def test_animate(ctx):
    """Validates the animation pipeline by testing all available presets"""
    try:
        status = await ctx.send("Starting animation system test...")
        
        for preset_name in EFFECT_PRESETS.keys():
            try:
                await status.edit(content=f"Testing preset: {preset_name}")
                
                processor = AnimationProcessor(INPUT_IMAGE)
                frames = processor.generate_frames(EFFECT_PRESETS[preset_name], num_frames=15)
                
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
        for effect in ['rgb', 'color', 'glitch', 'chroma', 'scan', 'noise', 'energy', 'pulse']:
            if effect not in params:
                continue
            processor.base_image = result.copy()
            result = processor.apply_effect(effect, params)
            # Generate title and tags based on effects used
            effect_tags = [effect for effect in ['rgb', 'color', 'glitch', 'chroma', 'scan', 'noise', 'energy', 'pulse'] 
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

@bot.command(name='remix')
async def remix_artwork(ctx, artwork_id: str, *args):
    """Remix existing artwork with new effects
    Usage: !remix <artwork_id> [effect parameters]"""
    try:
        # Get the original artwork
        original_image, metadata = art_repo.get_artwork(artwork_id)
        
        # Parse effect parameters
        params = parse_discord_args(args)
        
        # Process the image
        processor = ImageProcessor(original_image)
        result = processor.base_image
        
        # Apply effects in order
        for effect in EFFECT_ORDER:
            if effect in params:
                processor.base_image = result
                result = processor.apply_effect(effect, params)
        
        # Convert to bytes for storage
        output = BytesIO()
        result.save(output, format='PNG')
        output.seek(0)
        
        # Store remixed artwork
        new_artwork_id = art_repo.store_artwork(
            image=output.getvalue(),
            title=f"Remix of {metadata['title']}",
            creator_id=str(ctx.author.id),
            creator_name=ctx.author.name,
            parent_id=artwork_id,
            tags=['remix'] + list(params.keys()),
            parameters=params
        )
        
        # Send the remixed image
        output.seek(0)
        message = await ctx.send(
            f"Remixed artwork {artwork_id}! New ID: {new_artwork_id}\n"
            f"Original by: {metadata['creator_name']}",
            file=discord.File(fp=output, filename=f"remix_{new_artwork_id}.png")
        )
        
        # Add deletion reaction
        await message.add_reaction("🗑️")
        
    except Exception as e:
        await ctx.send(f"Error remixing artwork: {str(e)}")

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
            "--consciousness [0-1]"
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