import asyncio
import os
import sys
from pathlib import Path
import discord
from discord.ext import commands
from dotenv import load_dotenv

from image_processing.config.config import ConfigManager
from image_processing.storage.repository import ImageRepository
from image_processing.interface.command_parser import CommandParser
from image_processing.core.effect_processor import EffectProcessor
from image_processing.core.animation_processor import AnimationProcessor
from image_processing.core.ascii_processor import ASCIIProcessor

# Load environment variables
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')

# Initialize configuration
config = ConfigManager(config_dir="config")
repository = ImageRepository(db_path="image_repository.db", storage_path="image_storage")
command_parser = CommandParser(config)

# Set up Discord bot
intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True
bot = commands.Bot(command_prefix='!', intents=intents, help_command=None)

@bot.event
async def on_ready():
    """Bot startup event handler."""
    print(f'Image processing bot is online as {bot.user}')

@bot.event
async def on_reaction_add(reaction, user):
    """Handle reaction events for message cleanup."""
    if user != bot.user and str(reaction.emoji) == "🗑️":
        if reaction.message.author == bot.user:
            await reaction.message.delete()

@bot.command(name='process')
async def process_command(ctx, *args):
    """Process an image with effects."""
    try:
        # Parse command
        command = command_parser.parse_command(' '.join(args))
        
        # Get input image
        if ctx.message.attachments:
            attachment = ctx.message.attachments[0]
            image_bytes = await attachment.read()
            source = "uploaded image"
        else:
            if not Path("input.png").exists():
                await ctx.send("Please attach an image or ensure input.png exists!")
                return
            with open("input.png", "rb") as f:
                image_bytes = f.read()
            source = "input.png"

        # Process image
        processor = EffectProcessor(image_bytes)
        for effect_name, params in command.effects:
            processor.apply_effect(effect_name, params)

        # Save and send result
        output = processor.get_current_image()
        
        # Store in repository if configured
        if command.tags:
            image_id = repository.store_image(
                image=output,
                title=f"Processed_{ctx.author.name}",
                creator_id=str(ctx.author.id),
                creator_name=ctx.author.name,
                tags=command.tags,
                parameters=dict(command.effects)
            )
            await ctx.send(f"Image stored with ID: {image_id}")

        # Send processed image
        await ctx.send(file=discord.File(output, filename="processed.png"))

    except Exception as e:
        await ctx.send(f"Error processing image: {str(e)}")

@bot.command(name='animate')
async def animate_command(ctx, *args):
    """Create an animation with effects."""
    try:
        # Parse command
        command = command_parser.parse_command(f"animate {' '.join(args)}")
        
        # Get input image
        if ctx.message.attachments:
            attachment = ctx.message.attachments[0]
            image_bytes = await attachment.read()
        else:
            if not Path("input.png").exists():
                await ctx.send("Please attach an image or ensure input.png exists!")
                return
            with open("input.png", "rb") as f:
                image_bytes = f.read()

        # Create animation
        processor = AnimationProcessor(image_bytes)
        try:
            status_msg = await ctx.send("Generating animation...")
            
            frames = processor.generate_frames(
                effects=command.effects,
                num_frames=command.animation_params.get('frames', 30)
            )
            
            video_path = processor.create_video(
                frame_paths=frames,
                frame_rate=command.animation_params.get('fps', 24)
            )
            
            if video_path and video_path.exists():
                await ctx.send(file=discord.File(str(video_path)))
                if command.tags:
                    video_id = repository.store_image(
                        image=video_path.read_bytes(),
                        title=f"Animation_{ctx.author.name}",
                        creator_id=str(ctx.author.id),
                        creator_name=ctx.author.name,
                        tags=command.tags + ['animation'],
                        parameters=dict(command.effects)
                    )
                    await ctx.send(f"Animation stored with ID: {video_id}")
            else:
                await ctx.send("Failed to create animation")
            
            await status_msg.delete()
            
        finally:
            processor.cleanup()

    except Exception as e:
        await ctx.send(f"Error creating animation: {str(e)}")

@bot.command(name='ascii')
async def ascii_command(ctx, *args):
    """Create ASCII art from an image."""
    try:
        # Parse command
        command = command_parser.parse_command(f"ascii {' '.join(args)}")
        
        # Get input image
        if ctx.message.attachments:
            attachment = ctx.message.attachments[0]
            image_bytes = await attachment.read()
        else:
            if not Path("input.png").exists():
                await ctx.send("Please attach an image or ensure input.png exists!")
                return
            with open("input.png", "rb") as f:
                image_bytes = f.read()

        # Generate ASCII art
        processor = ASCIIProcessor(image_bytes)
        ascii_art = processor.convert_to_ascii(
            cols=command.ascii_params.get('cols', 80),
            scale=command.ascii_params.get('scale', 0.43),
            moreLevels=True
        )
        
        # Create and save both text and image versions
        ascii_image = processor.create_ascii_image(ascii_art)
        
        # Store results if tagged
        if command.tags:
            image_id = repository.store_image(
                image=ascii_image,
                title=f"ASCII_{ctx.author.name}",
                creator_id=str(ctx.author.id),
                creator_name=ctx.author.name,
                tags=command.tags + ['ascii'],
                parameters=command.ascii_params
            )
            await ctx.send(f"ASCII art stored with ID: {image_id}")

        # Send results
        await ctx.send(file=discord.File(ascii_image, filename="ascii.png"))
        await ctx.send(file=discord.File('\n'.join(ascii_art).encode(), filename="ascii.txt"))

    except Exception as e:
        await ctx.send(f"Error creating ASCII art: {str(e)}")

@bot.command(name='help')
async def help_command(ctx):
    """Show help information."""
    await ctx.send(command_parser.format_help())

@bot.command(name='examples')
async def examples_command(ctx):
    """Show example commands."""
    examples = command_parser.get_example_commands()
    await ctx.send("Example commands:\n" + "\n".join(examples))

def main():
    """Main entry point."""
    if not DISCORD_TOKEN:
        print("Error: DISCORD_TOKEN not found in .env file")
        sys.exit(1)

    # Create required directories
    Path("config").mkdir(exist_ok=True)
    Path("image_storage").mkdir(exist_ok=True)

    # Windows-specific event loop policy
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    print("Starting image processing bot...")
    bot.run(DISCORD_TOKEN)

if __name__ == "__main__":
    main()