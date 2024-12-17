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
from PIL import Image
from tetimi import ImageProcessor, EFFECT_PRESETS
from artrepo import ArtRepository
from anims import AnimationProcessor 
import io
from artrepo import ArtRepository


if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load environment variables
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
IMAGES_FOLDER = 'images'
INPUT_IMAGE = 'input.png'

# Set up Discord intents
intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True
bot = commands.Bot(command_prefix='!', intents=intents, help_command=None)

# Initialize the ArtRepository globally
art_repo = ArtRepository(db_path="art_repository.db", storage_path="art_storage")
    
def parse_discord_args(args):
    """Parse command arguments with enhanced animation support"""
    result = {}
    args = ' '.join(args)
    
    # Add animate flag handling
    if '--animate' in args:
        result['animate'] = True
        #only check for frames and fps if animate is true
        if '--frames' in args:
            frames_match = re.search(r'--frames\s+(\d+)', args)
            if frames_match:
                result['frames'] = min(int(frames_match.group(1)), 120)
            else:
                result['frames'] = 30
        if '--fps' in args:
            fps_match = re.search(r'--fps\s+(\d+)', args)
            if fps_match:
                result['fps'] = min(int(fps_match.group(1)), 60)  # Cap at 60fps
            else:
                result['fps'] = 24  # Default
    
    # Animation preset handling (separate from effect presets)
    anim_preset_match = re.search(r'--style\s+(\w+)', args)
    if anim_preset_match:
        preset_name = anim_preset_match.group(1).lower()
        if preset_name in ANIMATION_PRESETS:
            result['style'] = preset_name
        else:
            raise ValueError(f"Unknown animation preset. Available: {', '.join(ANIMATION_PRESETS.keys())}")
    

    
    # Alpha handling (existing code)
    alpha_match = re.search(r'--alpha\s+(\d+)', args)
    if alpha_match:
        alpha = int(alpha_match.group(1))
        if 0 <= alpha <= 255:
            result['alpha'] = alpha
        else:
            raise ValueError("Alpha value must be between 0 and 255")
    else:
        result['alpha'] = 180  # Default alpha value
        
    # Points effect handling (existing code)
    if '--points' in args:
        result['points'] = True
        dot_match = re.search(r'--dot-size\s+(\d+)', args)
        if dot_match:
            result['dot_size'] = int(dot_match.group(1))
        reg_match = re.search(r'--reg-offset\s+(\d+)', args)
        if reg_match:
            result['registration_offset'] = int(reg_match.group(1))
    
    # Random image handling (existing code)
    if '--random' in args:
        images = list(Path(IMAGES_FOLDER).glob('*.*'))
        if not images:
            raise ValueError(f"No images found in {IMAGES_FOLDER}")
        result['image_path'] = str(random.choice(images))
    
    # Effect preset handling (separate from animation presets)
    preset_match = re.search(r'--preset\s+(\w+)', args)
    if preset_match:
        preset_name = preset_match.group(1).lower()
        if preset_name in EFFECT_PRESETS:
            result.update(EFFECT_PRESETS[preset_name])
            return result
        raise ValueError(f"Unknown effect preset. Available: {', '.join(EFFECT_PRESETS.keys())}")
    
    # Color parameter handling (existing code)
    match = re.search(r'--(rgb|color)\s+(\d+)\s+(\d+)\s+(\d+)(?:\s+--\1alpha\s+(\d+))?', args)
    if match:
        color_type = match.group(1)
        r, g, b = map(int, match.groups()[1:4])
        alpha = int(match.group(5)) if match.group(5) else 255
        
        if all(0 <= x <= 255 for x in [r, g, b, alpha]):
            result[color_type] = (r, g, b)
            result[f'{color_type}alpha'] = alpha
        else:
            raise ValueError("Color/Alpha values must be between 0 and 255")
    
    # Effect parameter handling (existing code)
    params = {
        'glitch': (r'--glitch\s+(\d*\.?\d+)', lambda x: 1 <= float(x) <= 50, 
                  "Glitch intensity must be between 1 and 50"),
        'chroma': (r'--chroma\s+(\d*\.?\d+)', lambda x: 1 <= float(x) <= 40, 
                  "Chromatic aberration must be between 1 and 40"),
        'scan': (r'--scan\s+(\d*\.?\d+)', lambda x: 1 <= float(x) <= 200, 
                "Scan line gap must be between 1 and 200"),
        'noise': (r'--noise\s+(\d*\.?\d+)', lambda x: 0 <= float(x) <= 1, 
                 "Noise intensity must be between 0 and 1"),
        'energy': (r'--energy\s+(\d*\.?\d+)', lambda x: 0 <= float(x) <= 1, 
                  "Energy intensity must be between 0 and 1"),
        'pulse': (r'--pulse\s+(\d*\.?\d+)', lambda x: 0 <= float(x) <= 1, 
                 "Pulse intensity must be between 0 and 1")
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
        animate = params.pop('animate', False)
        image_path = params.pop('image_path', INPUT_IMAGE)

        if ctx.message.attachments:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                await ctx.message.attachments[0].save(tmp.name)
                image_path = tmp.name

        if not Path(image_path).exists():
            await ctx.send(f"Error: Image not found!")
            return

        # Process base image with effects
        processor = ImageProcessor(image_path)
        result = processor.base_image.convert('RGBA')

        # Apply effects in order
        for effect in ['rgb', 'color', 'glitch', 'chroma', 'scan', 'noise', 'energy', 'pulse']:
            if effect not in params:
                continue
            processor.base_image = result.copy()
            result = processor.apply_effect(effect, params)

        if animate:
            await ctx.send("Generating animation...")
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                result.save(tmp.name)
                anim = AnimationProcessor(tmp.name)
                
                frames = params.get('frames', 30)
                fps = params.get('fps', 24)
                style = params.get('style', 'glitch_surge')
                
                frames = anim.generate_frames(preset_name=style, num_frames=frames)
                video = anim.create_video(frame_rate=fps)
                
                if video and video.exists():
                    await ctx.send(file=discord.File(str(video)))
                else:
                    await ctx.send("Failed to generate animation")
                    
                os.unlink(tmp.name)
        else:
            output = BytesIO()
            result.save(output, format='PNG')
            output.seek(0)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_tetimi.png"
            message = await ctx.send(file=discord.File(fp=output, filename=filename))
            await message.add_reaction("🗑️")

    except Exception as e:
        await ctx.send(f"Error: {str(e)}")

@bot.command(name='testanimate')
async def test_animate(ctx):
    """Validates the animation pipeline with a sequence of effect tests"""
    try:
        status = await ctx.send("Starting animation system test...")
        
        processor = AnimationProcessor(INPUT_IMAGE)
        preset = 'glitch_surge'
        
        await status.edit(content=f"Testing preset: {preset}")
        frames = processor.generate_frames(preset_name='glitch_surge', num_frames=15)
        
        video_path = processor.create_video(
            frame_rate=24,
            output_name=f"test_{preset}.mp4"
        )
        
        if video_path and video_path.exists():
            await ctx.send(
                f"Preset {preset} test complete",
                file=discord.File(str(video_path))
            )
        else:
            await ctx.send(f"Failed to create video for {preset}")
            return
        
        await status.edit(content="Animation system test complete")
        
    except Exception as e:
        await ctx.send(f"Test failed: {str(e)}")
        return


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
            "--pulse [0-1]"
        ),
        inline=False
    )
    
    embed.add_field(
        name="Special Effects",
        value=(
            "--points - Apply points effect\n"
            "--dot-size [size] - Points dot size\n"
            "--reg-offset [offset] - Points registration offset\n"
            "--ascii - Convert to ASCII art\n"
            "--animate - Create animation (with --frames [n] --fps [n])"
        ),
        inline=False
    )

    await ctx.send(embed=embed)




@bot.command(name='store')
async def store_artwork(ctx, title: str = None, *tags):
    """Store artwork with an optional title and tags"""
    if not ctx.message.attachments:
        await ctx.send("Please attach an image!")
        return
    
    # If no title provided, use filename
    if not title:
        title = Path(ctx.message.attachments[0].filename).stem
        
    # If no tags provided, use title words as tags
    if not tags:
        tags = title.lower().split()
    
    artwork_id = art_repo.store_artwork(
        image=await ctx.message.attachments[0].read(),
        title=title,
        creator_id=str(ctx.author.id),
        creator_name=ctx.author.name,
        tags=list(tags)
    )
    await ctx.send(f"Artwork stored! ID: {artwork_id}")



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
        result = processor.convert_to_rgba()
        
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




@bot.command(name='remix')
async def remix_artwork(ctx, artwork_id: str, *args):
    """Apply effects to existing artwork"""
    try:
        # Fetch the original artwork from the repository
        original_image, metadata = art_repo.get_artwork(artwork_id)

        # Initialize the ImageProcessor with the original artwork
        processor = ImageProcessor(original_image)

        # Parse arguments to identify which effects to apply
        params = parse_discord_args(args)

        # Check if there's an attachment to use as the remix base image
        if ctx.message.attachments:
            attachment = ctx.message.attachments[0]
            image_bytes = await attachment.read()
            base_image = Image.open(io.BytesIO(image_bytes))  # Open the image from bytes
            processor.base_image = base_image  # Set the base image to the attachment
            
        # Process the image (apply effects)
        result = processor.convert_to_rgba()  # Start with the original image (converted to RGBA)

        # Apply effects in order
        for effect in ['rgb', 'color', 'glitch', 'chroma', 'scan', 'noise', 'energy', 'pulse']:
            if effect not in params:
                continue
            processor.base_image = result.copy()
            result = processor.apply_effect(effect, params)

        # Store the remixed image
        new_artwork_id = art_repo.store_artwork(
            image=result,
            title=f"Remix_{metadata['title']}",
            creator=str(ctx.author.id),
            parent_id=artwork_id,
            tags=['remix'],
            parameters=params
        )

        # Save the processed image to a BytesIO object
        output = io.BytesIO()
        result.save(output, format='PNG')
        output.seek(0)

        # Send the remixed image back to the user
        await ctx.send(
            f"Artwork remixed! New ID: {new_artwork_id}\nOriginal by: <@{metadata['creator']}>",
            file=discord.File(fp=output, filename=f"{new_artwork_id}.png")
        )

    except Exception as e:
        await ctx.send(f"Error remixing artwork: {str(e)}")


@bot.command(name='search')
async def search_artwork(ctx, *, query: str = ""):
    try:
        results = art_repo.search_artwork(query)
        
        if not results:
            await ctx.send("No artwork found matching those terms.")
            return
            
        embed = discord.Embed(title="Artwork Search Results", color=discord.Color.purple())
        
        for art in results:
            embed.add_field(
                name=f"{art['title']} (ID: {art['id']})",
                value=f"Creator: {art['creator_name']}\nTags: {', '.join(art['tags'])}",
                inline=False
            )
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"Error searching artwork: {str(e)}")


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
                value=f"Creator: {creator.name}\nInteractions: {artwork['interactions']}",
                inline=False
            )
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"Error getting trending artwork: {str(e)}")


@bot.command(name='history')
async def show_history(ctx, artwork_id: str):
    """Show modification history of artwork"""
    try:
        history = art_repo.get_artwork_history(artwork_id)
        original, metadata = art_repo.get_artwork(artwork_id)
        
        embed = discord.Embed(
            title=f"History for {metadata['title']}",
            description=f"Original creator: <@{metadata['creator']}>",
            color=discord.Color.purple()
        )
        
        for version in history:
            modifier = await bot.fetch_user(int(version['modifier']))
            embed.add_field(
                name=f"Version {version['version']}",
                value=f"Modified by: {modifier.name}\nTimestamp: <t:{int(version['timestamp'])}:R>",
                inline=False
            )
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"Error getting artwork history: {str(e)}")

@bot.event
async def on_shutdown():
    art_repo.conn.close()
    print("Database connection closed.")


def main():
    if not DISCORD_TOKEN:
        print("Error: DISCORD_TOKEN not found in .env file")
        return
    print("Starting Tetimi bot...")
    bot.run(DISCORD_TOKEN)
    bot.load_extension('art_commands').txt

if __name__ == "__main__":
    main()