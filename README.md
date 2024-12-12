# Tetsuo Image Effect Bot

Discord bot that applies Akira/Tetsuo-inspired visual effects to images with adaptive processing.

## Features

- Automatic effect adaptation based on image analysis
- RGB color overlay with smart blending
- Chromatic aberration and glitch effects
- Scanline generation with variable intensity
- Multiple preset styles (cyberpunk, vaporwave, akira, etc.)
- Random image selection from folder
- Image validation and safety checks

## Commands

```
!tetsuo [options]
!tetsuo_presets - List available presets
!tetsuo_help - Show help message
```

### Options

- `--preset <name>` - Use predefined effect combination
- `--rgb R G B` - Custom RGB color values (0-255)
- `--alpha` - Transparency (0-255)
- `--chroma` - Chromatic aberration (1-20)
- `--glitch` - Glitch effect intensity (1-20)
- `--scan` - Scanline gap (1-10)
- `--noise` - Noise intensity (0.0-1.0)
- `--random` - Select random image from images folder

### Examples

```
!tetsuo --preset akira --random
!tetsuo --rgb 255 0 0 --glitch 10
!tetsuo --preset cyberpunk
```

## Installation

1. Clone repository
2. Install requirements: `pip install -r requirements.txt`
3. Create `.env` file with `DISCORD_TOKEN=your_token`
4. Create `images` folder for input images
5. Run: `python tetimi.py`

## Image Requirements

- Formats: PNG, JPEG
- Size: 50x50 to 2000x2000 pixels
- Max file size: 8MB

## Effect Parameters

Default user/adaptive weight ratio: 0.6/0.4
- Brightness impacts alpha strength
- Image complexity affects glitch intensity
- Color variance influences chromatic aberration
- Scan lines adapt to image detail level

## Presets

- `akira`: Red dominant, medium glitch
- `tetsuo`: Purple dominant, high glitch
- `cyberpunk`: Cyan accent, low noise
- `vaporwave`: Pink accent, high chroma
- `matrix`: Green dominant, light scan lines
- `synthwave`: Magenta accent, medium effects

## License

MIT License
