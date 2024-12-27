import asyncio
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import discord
from discord.ext import commands
from dotenv import load_dotenv

from ..config.config import ConfigManager
from ..core.effect_processor import EffectProcessor
from ..core.image_processor import BaseImageProcessor
from ..effects.animation_effects import AnimationProcessor, ASCIIProcessor
from ..storage.repository import ImageRepository
from .command_parser import CommandParser, ParsedCommand

# Load environment variables
# Initialize configuration
load_dotenv()
configs = ConfigManager(config_dir="config")
DISCORD_TOKEN = configs.DISCORD_TOKEN

repository = ImageRepository(
    db_path="image_repository.db", storage_path="image_storage"
)
command_parser = CommandParser(configs)

# Set up Discord bot
intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)


@bot.event
async def on_ready():
    """Bot startup event handler."""
    print(f"Image processing bot is online as {bot.user}")


@bot.event
async def on_reaction_add(reaction, user):
    """Handle reaction events for message cleanup."""
    if user != bot.user and str(reaction.emoji) == "🗑️":
        if reaction.message.author == bot.user:
            await reaction.message.delete()

'''
@bot.command(name="image")
async def image_command(ctx, *args):
    """Process an image with effects."""
    try:
        # Parse command
        command = await command_parser.parse_command(ctx, f"process {' '.join(args)}")

        await ctx.send("Command returned {type(command)}")

        # Get input image
        if ctx.message.attachments:
            attachment = ctx.message.attachments[0]
            image_bytes = await attachment.read()

        # Process image
    except Exception as e:
        await ctx.send(f"Error selecting image, process_command: {str(e)}")
        return

    try:
        my_params = {}
        processor = BaseImageProcessor(image_bytes)
        e_processor = EffectProcessor(processor.original_image)
        if len(command.effects) > 0:
            for effect_name, params in command.effects:
                e_processor.apply_effect(effect_name, params)
                my_params[effect_name] = params

        # Save and send result
        my_image = e_processor.get_current_image()

    except Exception as e:
        await ctx.send(f"Error processings Effects, {str(e)}")

    try:
        # Store in repository if configured
        if command.tags:
            image_id = repository.store_image(
                image=my_image,
                title=f"Processed_{ctx.author.name}",
                creator_id=str(ctx.author.id),
                creator_name=ctx.author.name,
                tags=command.tags,
                parameters=[None if len(my_params.keys()) == 0 else my_params],
            )
            await ctx.send(f"Image stored with ID: {image_id}")

        with BytesIO() as image_binary:
            my_image.save(image_binary, "PNG")
            image_binary.seek(0)
        # Send processed image
        file = discord.File(fp=image_binary, filename="processed.png")

        await ctx.send(file=file)

    except Exception as e:
        await ctx.send(f"Error processing image, process_command: {str(e)}")
'''

@bot.command(name="image")
async def image_command(ctx, *args):
    """Process image with effects and optional animation"""
    try:
        # Get image input
        image_input = await _get_image_input(ctx)
        if not image_input:
            image_input = None

        # Parse command
        try:
            parsed = await command_parser.parse_command(
                ctx, f"image {' '.join(args)}", image_input
            )
        except ValueError as e:
            await ctx.send(f"Invalid command: {str(e)}")
            return

        if parsed.command == "animate":
            await _handle_animation(ctx, parsed)
        else:
            await _handle_static_image(ctx, parsed)

    except Exception as e:
        # self.logger.error(f"Error processing image command: {str(e)}", exc_info=True)
        await ctx.send(f"Error processing image, image_command: {str(e)}")


async def _get_image_input(ctx) -> Optional[Union[str, Path]]:
    """Get image input from message attachment or local file"""
    if ctx.message.attachments:
        # Save attachment to temporary file
        attachment = ctx.message.attachments[0]
        image_data = await attachment.read()
        temp_path = Path("temp_input.png")
        temp_path.write_bytes(image_data)

        return str(temp_path)

    # Check for local input.png
    if Path("input.png").exists():
        return "input.png"

    # Will use random image if specified in command args
    return None


async def _handle_animation(ctx, parsed: ParsedCommand):
    """Handle animation generation and sending"""
    status_msg = await ctx.send("Generating animation...")

    try:
        # Initialize processor
        if parsed.image_path is None:
            parsed.image_path = "input.png"

        processor = EffectProcessor(parsed.image_path)
        anim_processor = AnimationProcessor(processor.base_image)

        # Generate frames
        frames = anim_processor.generate_frames(
            effects=parsed.effects,
            num_frames=parsed.animation_params.get("frames", 30),
        )

        # Create video
        video_path = anim_processor.create_video(
            frame_paths=frames,
            frame_rate=parsed.animation_params.get("fps", 24),
            output_path="animation.mp4",
        )

        if video_path and video_path.exists():
            await status_msg.edit(content="Animation complete!")
            await ctx.send(file=discord.File(str(video_path)))
        else:
            await status_msg.edit(content="Failed to create animation")

    except Exception as e:
        # self.logger.error(f"Error creating animation: {str(e)}", exc_info=True)
        await status_msg.edit(content=f"Error creating animation: {str(e)}")
    finally:
        if "anim_processor" in locals():
            anim_processor.cleanup()


async def _handle_static_image(ctx, parsed: ParsedCommand):
    """Handle static image processing and sending"""
    processor = BaseImageProcessor(
        parsed.image_path, "random" in vars(parsed.output_params)
    )

    e_processor = EffectProcessor(processor.original_image)

    # Apply all effects in order
    for effect, params in parsed.effects:
        processor.current_image = e_processor.apply_effect(effect, params)

        # Apply output parameters
    output_format = parsed.output_params.get("format", "PNG")
    quality = parsed.output_params.get("quality", 95)

    # Save and send
    buffer = BytesIO()
    processor.base_image.save(
        buffer,
        format=output_format,
        quality=quality,
        **{
            k: v
            for k, v in parsed.output_params.items()
            if k in ["alpha", "coloralpha", "rgbalpha"]
        },
    )
    buffer.seek(0)

    filename = f"processed.{output_format.lower()}"
    await ctx.send(
        file=discord.File(buffer, filename),
        content=f"Tags: {', '.join(parsed.tags)}" if parsed.tags else None,
    )


@bot.command(name="animate")
async def animate_command(ctx, *args):
    """Create an animation with effects."""
    try:
        # Parse command
        command = await command_parser.parse_command(ctx, f"animate {' '.join(args)}")

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
                num_frames=command.animation_params.get("frames", 30),
            )

            video_path = processor.create_video(
                frame_paths=frames, frame_rate=command.animation_params.get("fps", 24)
            )

            if video_path and video_path.exists():
                await ctx.send(file=discord.File(str(video_path)))
                if command.tags:
                    video_id = repository.store_image(
                        image=video_path.read_bytes(),
                        title=f"Animation_{ctx.author.name}",
                        creator_id=str(ctx.author.id),
                        creator_name=ctx.author.name,
                        tags=command.tags + ["animation"],
                        parameters=dict(command.effects),
                    )
                    await ctx.send(f"Animation stored with ID: {video_id}")
            else:
                await ctx.send("Failed to create animation")

            await status_msg.delete()

        finally:
            processor.cleanup()

    except Exception as e:
        await ctx.send(f"Error creating animation: {str(e)}")


@bot.command(name="ascii")
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
            cols=command.ascii_params.get("cols", 80),
            scale=command.ascii_params.get("scale", 0.43),
            moreLevels=True,
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
                tags=command.tags + ["ascii"],
                parameters=command.ascii_params,
            )
            await ctx.send(f"ASCII art stored with ID: {image_id}")

        # Send results
        await ctx.send(file=discord.File(ascii_image, filename="ascii.png"))
        await ctx.send(
            file=discord.File("\n".join(ascii_art).encode(), filename="ascii.txt")
        )

    except Exception as e:
        await ctx.send(f"Error creating ASCII art: {str(e)}")


@bot.command(name="help")
async def help_command(ctx):
    """Show help information."""
    await ctx.send(command_parser.format_help())


@bot.command(name="examples")
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
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    print("Starting image processing bot...")
    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
