import asyncio
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import discord
from discord.ext import commands
from dotenv import load_dotenv

from PIL import Image
from ..config.config import ConfigManager, AnimationConfig, DISCORD_TOKEN
from ..core.effect_processor import EffectProcessor
from ..effects.animation_effects import AnimationProcessor, ASCIIProcessor
from ..storage.repository import ImageRepository
from .command_parser import CommandParser, ParsedCommand

# Load environment variables
# Initialize configuration
load_dotenv()
configs = ConfigManager(config_dir="config")

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
            print(f"args = {args}\njoin_args = {' '.join(args)}")
            parsed = await command_parser.parse_command(
                ctx, f"image {' '.join(args)}", image_input
            )
            await ctx.send(f"{parsed}")
        except ValueError as e:
            await ctx.send(f"Invalid command: {str(e)}")
            return

        if "--animate" in parsed.effects:
            await ctx.send(f"{parsed.effects.items()}")
            await _handle_animation(ctx, parsed)
        else:
            await ctx.send(f"{parsed.effects.items()}")
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
            parsed.image_path = configs.INPUT_IMAGE

        processor = EffectProcessor(parsed.image_path)
        anim_processor = AnimationProcessor(processor.base_image)

        try:
            fps = parsed.effects['fps']['count']
        except KeyError:
            fps = AnimationConfig.default_fps

        # Generate frames
        frames = anim_processor.generate_frames(
            effects=parsed.effects,
            num_frames=fps,
        )

        # Create video
        video_path = anim_processor.create_video(
            frame_paths=frames,
            frame_rate=fps,
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

    load_image = configs.INPUT_IMAGE if parsed.image_path is None else parsed.image_path

    e_processor = EffectProcessor(load_image)

    # Apply all effects in order
    print(parsed.effects)

    e_processor.apply_effects_sequence(parsed.effects)

    # Apply output parameters
    '''
    output_format = parsed.output_params.get("format", "PNG")
    quality = parsed.output_params.get("quality", 95)
    '''

    # Temporary until output_params is implemented
    output_format = "PNG"
    quality = 95

    # Save and send
    buffer = BytesIO()
    e_processor.current_image.save(
        buffer,
        format=output_format,
        quality=quality,
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

        try:
            frames = command.effects['frames']['count']
        except KeyError:
            frames = AnimationConfig.default_frames

        try:
            fps = command.effects['fps']['count']
        except KeyError:
            fps = AnimationConfig.default_fps

        # Create animation
        processor = AnimationProcessor(image_bytes)
        try:
            status_msg = await ctx.send("Generating animation...")

            frames = processor.generate_frames(
                effects=command.effects,
                num_frames=frames,
            )

            video_path = processor.create_video(
                frame_paths=frames, frame_rate=fps
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
        print(f"Args: {args}")
        if len(args) > 1:
            parsed: ParsedCommand = await command_parser.parse_command(
                    ctx, f"ascii {' '.join(args)}")

        else:
            parsed: ParsedCommand = await command_parser.parse_command(
                    ctx, "ascii")

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

        if parsed.ascii_params is None:
            ctx.send("No ASCII parameters specified")
            return

        image = Image.open(BytesIO(image_bytes))

        print("Got to Processor")

        # Generate ASCII art
        try:
            cols=parsed.ascii_params.cols
        except Exception:
            cols=configs.ascii.default_cols

        try:
            scale=parsed.ascii_params.cols
        except Exception:
            scale=configs.ascii.default_scale
        processor = ASCIIProcessor(BytesIO(image_bytes))
        ascii_image = \
            processor.convert_to_ascii_image(image,
                                             cols=cols,
                                             scale=scale,
                                             moreLevels=True,
                                             )

        # Create and save both text and image versions
        # ascii_image = processor.create_gif(ascii_art)

        # Store results if tagged
        if parsed.tags:
            image_id = repository.store_image(
                image=ascii_image,
                title=f"ASCII_{ctx.author.name}",
                creator_id=str(ctx.author.id),
                creator_name=ctx.author.name,
                tags=parsed.tags + ["ascii"],
                parameters=parsed.ascii_params,
            )
            await ctx.send(f"ASCII art stored with ID: {image_id}")

        # Send results
        ascii_image.save("ascii_output.png")
        await ctx.send(file=discord.File("ascii_output.png",
                                         filename="ascii.png"))

    except Exception as e:
        await ctx.send(f"Error creating ASCII art: {str(e)}")


@bot.command(name="help")
async def help_command(ctx):
    """Show help information."""
    await ctx.send(await command_parser.format_help(ctx))


@bot.command(name="examples")
async def examples_command(ctx):
    """Show example commands."""
    # examples = command_parser.get_example_commands()
    await ctx.send("Example commands: TBD\n")  # + "\n".join(examples))


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
