import os
import shutil
from io import BytesIO
from typing import Union, AsyncGenerator
import logging
from PIL import Image, ImageDraw, ImageFont
from PIL.Image import Resampling

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse

# from pydantic import BaseModel

DENIED = "denied.png"

app = FastAPI()

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Get a logger instance (you can name your logger for better organization)
log = logging.getLogger(__name__)

# Define a Pydantic model for request body validation


class ImpactParameters:
    text: str = "$TETSUO"

    def __init__(self, text: str):
        self.text = text


@app.get("/watermark/upload", response_class=HTMLResponse)
async def watermark_redirect_to_post():
    # This GET endpoint will redirect to the POST endpoint
    return """
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>File Upload</title>
    </head>
    <body>
        <form action="/watermark" method="post" enctype="multipart/form-data">
            <label for="file">Choose file to upload:</label>
            <input type="file" id="file" name="file" required>
            <br>
            <input type="submit" value="Upload">
        </form>
    </body>
    </html>
    """


@app.get("/denied/upload", response_class=HTMLResponse)
async def denied_redirect_to_post():
    # This GET endpoint will redirect to the POST endpoint
    return """
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>File Upload</title>
    </head>
    <body>
        <form action="/denied" method="post" enctype="multipart/form-data">
            <label for="file">Choose file to upload:</label>
            <input type="file" id="file" name="file" required>
            <br>
            <input type="submit" value="Upload">
        </form>
    </body>
    </html>
    """


@app.get("/impact/upload/{text}", response_class=HTMLResponse)
async def redirect_to_post(text: str = ""):
    # This GET endpoint will redirect to the POST endpoint
    return f"""
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>File Upload</title>
    </head>
    <body>
        <form action="/impact/{text}" method="post" enctype="multipart/form-data">
            <label for="file">Choose file to upload:</label>
            <input type="file" id="file" name="file" required>
            <br>
            <input type="submit" value="Upload">
        </form>
    </body>
    </html>
    """


async def byte_array_generator(byte_array: bytes) -> AsyncGenerator[bytes, None]:
    yield byte_array


@app.post("/watermark")
async def watermark(file: UploadFile = File(...), response_class=HTMLResponse):
    try:
        with open(f"{file.filename}", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if file.filename is None:
            file.filename = "test.png"

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: str({e})")

    try:
        watermark_service = WatermarkEffectService()

        finished_image = await watermark_service.watermark_frame(
            Image.open(file.filename)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"WatermarkEffect gen failed: str({e})"
        )
    output = BytesIO()
    finished_image.save(output, format="PNG")

    # FUCK OFF
    if output is None:
        raise HTTPException(status_code=500, detail="Output generation broke. str{e}")

    output.seek(0)

    # Write the processed image to a new file
    new_filename = f"processed_{file.filename}"
    with open(new_filename, "wb") as new_file:
        new_file.write(output.getvalue())

    html_content = f"""
    <!DOCTYPE html>
        <html>
            <head>
                <title>Watermark Generator: {file.filename}</title>
            </head>
        <body>
            w<img id="img" src="/serve_image/{new_filename}" alt="$TETSUO"/>
        </body>
        </html>
    """.strip()

    return HTMLResponse(content=html_content, status_code=200)


@app.post("/denied")
async def denied(file: UploadFile = File(...), response_class=HTMLResponse):
    try:
        with open(f"{file.filename}", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if file.filename is None:
            file.filename = "test.png"

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: str({e})")

    try:
        denied_service = DeniedEffectService()
        finished_image = await denied_service.denied_frame(Image.open(file.filename))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Effect generation failed: str({e})"
        )
    output = BytesIO()
    finished_image.save(output, format="PNG")

    # FUCK OFF
    if output is None:
        raise HTTPException(status_code=500, detail="Output generation broke. str{e}")

    output.seek(0)

    # Write the processed image to a new file
    new_filename = f"processed_{file.filename}"
    with open(new_filename, "wb") as new_file:
        new_file.write(output.getvalue())

    html_content = f"""
        <!DOCTYPE html>
            <html>
                <head>
                    <title>DENIED Generator: {file.filename}</title>
                </head>
            <body>
                <img id="img" src="/serve_image/{new_filename}" alt="Image from Byte Array"/>
            </body>
            </html>
    """.strip()

    return HTMLResponse(content=html_content, status_code=200)


# Setup base endpoint for IMPACT effect
@app.post("/impact/{text}")
async def impact(
    file: UploadFile = File(...), text: str = "$TETSUO", response_class=HTMLResponse
):

    try:

        with open(f"{file.filename}", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        params = ImpactParameters("$TEST")
        params.text = text

        if file.filename is None:
            file.filename = "test.png"

        impact_service = ImpactEffectService()
        finished_image = await impact_service.generate_frame(
            Image.open(file.filename), params
        )
        output = BytesIO()
        finished_image.save(output, format="PNG")
        output = output.getvalue()
        print("received call")

        if file.filename is None:
            raise HTTPException(status_code=500, detail="No file uploaded")

    except Exception as e:
        log.exception(f"Impact text addition failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    os.remove(file.filename)

    output = BytesIO()
    finished_image.save(output, format="PNG")
    output.seek(0)

    # Write the processed image to a new file
    new_filename = f"processed_{file.filename}"
    with open(new_filename, "wb") as new_file:
        new_file.write(output.getvalue())

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
<title>IMPACT Generator: {file.filename}</title>
</head>
<body>
<img id="img" src="/serve_image/{new_filename}" alt="Image from Byte Array"/>
</body>
</html>
""".strip()

    return HTMLResponse(content=html_content, status_code=200)


class WatermarkEffectService:
    """Service for generating RGB channel pass animations.
    Direct port of Discord bot's ChannelPassAnimator functionality."""

    def __init__(self):
        super().__init__()

    async def start(self) -> None:
        """Initialize and start the service"""
        log.info("Starting DeniedEffectService...")

    async def stop(self) -> None:
        """Cleanup and stop the service"""
        log.info("Stopping DeniedtEffectService...")

    def ensure_even_dimensions(self, image: Image.Image) -> Image.Image:
        """Ensure image dimensions are even"""
        width, height = image.size
        new_width = width if width % 2 == 0 else width + 1
        new_height = height if height % 2 == 0 else height + 1

        if new_width != width or new_height != height:
            new_image = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))
            new_image.paste(
                image, ((new_width - width) // 2, (new_height - height) // 2)
            )
            return new_image
        return image

    async def watermark_frame(
        self, image: Union[bytes, Image.Image, BytesIO]
    ) -> Image.Image:
        """Generate frames with channel pass effect"""
        try:
            # Convert input to PIL Image
            if isinstance(image, bytes):
                base_image = Image.open(BytesIO(image))
            elif isinstance(image, Image.Image):
                base_image = image
            elif isinstance(image, BytesIO):
                base_image = Image.open(image)
            else:
                raise ValueError("Unsupported image input type")

            # Ensure even dimensions
            base_image = self.ensure_even_dimensions(base_image).convert("RGBA")
        except Exception as e:
            log.info("Image parsing for Impact failed")
            raise HTTPException(status_code=500, detail=f"Error in parsing image: {e}")

        watermark_path = "tetsuo_rage.png"
        # Open the watermark image
        with Image.open(watermark_path) as watermark:
            # Resize watermark if needed (e.g., to 10% of base image's smaller dimension)
            scale = 0.1  # 10%
            watermark = watermark.convert("RGBA")
            watermark = watermark.resize(
                (int(base_image.width * scale), int(base_image.height * scale)),
                Resampling.LANCZOS,
            )

            # Create a new transparent layer for compositing
            watermark_layer = Image.new("RGBA", base_image.size, (0, 0, 0, 0))

            # Calculate position for watermark (e.g., lower right corner with margin)
            margin = int(min(base_image.size) * 0.05)  # 1% margin
            position = (
                base_image.width - watermark.width - margin,
                base_image.height - watermark.height - margin,
            )

            # Paste the watermark onto the transparent layer, using its own alpha for blending
            watermark_layer.paste(watermark, position, watermark)

            # Reduce opacity of watermark to 27.5%
            new_alpha = int(255 * 0.235)
            watermark_layer.putalpha(new_alpha)
            # Composite base image with watermark
            out = Image.alpha_composite(base_image, watermark_layer)

        # Return the composited image as bytes
        output = BytesIO()
        out.save(output, format="PNG")
        output.seek(0)
        """
        alpha_image = Image.new("RGBA", watermark_image.size, (0, 0, 0, 0))
        alpha_image.paste(watermark_image, (0, 0), mask=watermark_image)
        alpha_image.putalpha(int(255 * 0.3))
        alpha_image = alpha_image.resize(
            (int(base_image.height * 0.1), int(base_image.height * 0.1)),
            Resampling.BILINEAR,
        )

        canvas = Image.new("RGBA", base_image.size)

        # Calculate the position to center the overlay
        overlay_width, _ = watermark_image.size
        background_width, background_height = base_image.size

        # Center position calculation
        x_offset = y_offset = (
            background_width
            - overlay_width
            - min(background_width, background_height) * 0.01
        )

        # Paste the smaller image onto this canvas at an appropriate position
        canvas.paste(alpha_image, (int(x_offset), int(y_offset)), alpha_image)
        """
        # Combine the text overlay with the original impact_image
        return out


class DeniedEffectService:
    """Service for generating RGB channel pass animations.
    Direct port of Discord bot's ChannelPassAnimator functionality."""

    def __init__(self):
        super().__init__()

    async def start(self) -> None:
        """Initialize and start the service"""
        log.info("Starting DeniedEffectService...")

    async def stop(self) -> None:
        """Cleanup and stop the service"""
        log.info("Stopping DeniedtEffectService...")

    def ensure_even_dimensions(self, image: Image.Image) -> Image.Image:
        """Ensure image dimensions are even"""
        width, height = image.size
        new_width = width if width % 2 == 0 else width + 1
        new_height = height if height % 2 == 0 else height + 1

        if new_width != width or new_height != height:
            new_image = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))
            new_image.paste(
                image, ((new_width - width) // 2, (new_height - height) // 2)
            )
            return new_image
        return image

    async def denied_frame(
        self, image: Union[bytes, Image.Image, BytesIO]
    ) -> Image.Image:
        """Generate frames with channel pass effect"""
        try:
            # Convert input to PIL Image
            if isinstance(image, bytes):
                base_image = Image.open(BytesIO(image))
            elif isinstance(image, Image.Image):
                base_image = image
            elif isinstance(image, BytesIO):
                base_image = Image.open(image)
            else:
                raise ValueError("Unsupported image input type")

            # Ensure even dimensions
            base_image = self.ensure_even_dimensions(base_image).convert("RGBA")
        except Exception as e:
            log.info("Image parsing for Impact failed")
            raise HTTPException(status_code=500, detail=f"Error in parsing image: {e}")

        denied_image = Image.open("denied.png").convert("RGBA")

        bigger_image = max([base_image, denied_image], key=lambda img: img.size)
        canvas = Image.new("RGBA", bigger_image.size)

        # Calculate the position to center the overlay
        overlay_width, overlay_height = denied_image.size
        background_width, background_height = base_image.size

        # Center position calculation
        x_offset = (background_width - overlay_width) // 2
        y_offset = (background_height - overlay_height) // 2

        # Decide how you want to scale the overlay.
        # Here's an example where the overlay would be resized to 50% of the background's dimensions:
        scale_factor = 0.5
        new_width = int(background_width * scale_factor)
        new_height = int(background_height * scale_factor)

        # Resize the overlay image
        denied_image = denied_image.resize((new_width, new_height))

        # Paste the smaller image onto this canvas at an appropriate position
        smaller_image = min([base_image, denied_image], key=lambda img: img.size)
        canvas.paste(
            smaller_image, (int(x_offset / 2), int(y_offset / 2)), smaller_image
        )

        # Combine the text overlay with the original impact_image
        try:
            combined = Image.alpha_composite(bigger_image, canvas)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in final combine: {e}")
        return combined


@app.get("/serve_image/{filename}")
async def serve_image(filename: str):

    if os.path.exists(filename):
        return FileResponse(filename, media_type="image/png")
    else:
        raise HTTPException(status_code=404, detail="File not found")


class ImpactEffectService:
    """Service for generating RGB channel pass animations.
    Direct port of Discord bot's ChannelPassAnimator functionality."""

    def __init__(self):
        super().__init__()

    async def start(self) -> None:
        """Initialize and start the service"""
        log.info("Starting ImpactEffectService...")

    async def stop(self) -> None:
        """Cleanup and stop the service"""
        log.info("Stopping ImpactEffectService...")

    def ensure_even_dimensions(self, image: Image.Image) -> Image.Image:
        """Ensure image dimensions are even"""
        width, height = image.size
        new_width = width if width % 2 == 0 else width + 1
        new_height = height if height % 2 == 0 else height + 1

        if new_width != width or new_height != height:
            new_image = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))
            new_image.paste(
                image, ((new_width - width) // 2, (new_height - height) // 2)
            )
            return new_image
        return image

    async def generate_frame(
        self, image: Union[bytes, Image.Image, BytesIO], params: ImpactParameters
    ) -> Image.Image:
        """Generate frames with channel pass effect"""
        try:
            # Convert input to PIL Image
            if isinstance(image, bytes):
                base_image = Image.open(BytesIO(image))
            elif isinstance(image, Image.Image):
                base_image = image
            elif isinstance(image, BytesIO):
                base_image = Image.open(image)
            else:
                raise ValueError("Unsupported image input type")

            # Ensure even dimensions
            base_image = self.ensure_even_dimensions(base_image)
        except Exception as _:
            log.info("Image parsing for Impact failed")
            raise ValueError("Error parsing image dimensions.")

        impact_image = base_image.copy().convert("RGBA")
        # Create a new image for the text overlay with transparency
        txt = Image.new("RGBA", impact_image.size, (255, 255, 255, 0))
        # Grab width for scaling
        width, height = impact_image.size
        # Load the font
        font = ImageFont.truetype("impact.ttf", int(height * 0.2))
        # Setup text for writing
        text = params.text
        if text == "":
            text = "$TETSUO"
        # Draw context
        d = ImageDraw.Draw(txt)
        # Define the text and colors
        outline_color = (0, 0, 0, 255)  # Black with full opacity
        text_color = (255, 255, 255, 255)  # White with full opacity
        # Calculate text size to center it
        text_bbox = d.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        position = (
            (width - text_width) // 2,
            (height - text_height) // 1.15,
        )
        # Draw outline by drawing the text shifted slightly in all directions
        for outline_offset in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            d.text(
                (position[0] + outline_offset[0], position[1] + outline_offset[1]),
                text,
                fill=outline_color,
                font=font,
            )
            # Draw the main text over the outline
            d.text(position, text, fill=text_color, font=font)
        # Combine the text overlay with the original impact_image
        combined = Image.alpha_composite(impact_image, txt)
        return combined

    async def get_status(self) -> dict:
        """Get service status and metrics"""
        return {
            "status": "online",
        }
