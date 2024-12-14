# Tetimi - Tetsuo Image Effect Bot

A Discord bot that creates cyberpunk-inspired visual effects inspired by Akira and Tetsuo. The bot uses adaptive processing to analyze images and apply effects that complement the original content.

## Features

- Image analysis system that adapts effects based on content
- Multiple effect types including RGB overlays, glitch effects, and chromatic aberration
- Sophisticated visual effects including:
  - Energy patterns with controllable intensity
  - Pulse effects with adaptive brightness
  - Enhanced chromatic aberration with color preservation
  - Advanced glitch effects with controlled randomization
  - Scanline generation with variable density
  - Noise effects with color relationship preservation
- ASCII art generation
- Silkscreen effect inspired by pop art
- Multiple preset combinations for different aesthetic styles
- Support for random image selection from a designated folder

## Commands

### Basic Usage
```
!image [options] - Process the default or specified image
!image --random - Process a random image from the images folder
!image --ascii - Generate ASCII art version
!image_help - Display help information
```

### Effect Options
- `--preset <name>` - Use a predefined effect combination
- `--rgb <r> <g> <b>` - Set RGB color values (0-255)
- `--alpha <value>` - Set transparency (0-255)
- `--glitch <1-50>` - Set glitch effect intensity
- `--chroma <1-40>` - Set chromatic aberration strength
- `--scan <1-200>` - Set scanline gap
- `--noise <0-2>` - Set noise intensity
- `--energy <0-2>` - Add energy effect with specified intensity
- `--pulse <0-2>` - Add pulse effect with specified intensity
- `--silkscreen` - Apply silkscreen effect

### Available Presets
- `cyberpunk`: Cyan-dominant with subtle glitch effects
- `vaporwave`: Pink-heavy with strong chromatic aberration
- `glitch_art`: Heavy distortion effects
- `retro`: Classic scanline look
- `matrix`: Green-tinted cyberpunk aesthetic
- `synthwave`: 80s-inspired magenta effects
- `akira`: Classic Akira-inspired red tones
- `tetsuo`: Intense purple with strong effects
- `neo_tokyo`: Urban cyberpunk aesthetic
- `psychic`: Psychedelic effect combination
- `tetsuo_rage`: Maximum intensity effects

## Installation

1. Clone the repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file containing:
   ```
   DISCORD_TOKEN=your_token_here
   ```
4. Create an `images` folder for input images
5. Run the bot:
   ```bash
   python tetimi.py
   ```

## Effect System

The bot uses an adaptive processing system that analyzes images to optimize effect parameters:
- Image brightness influences opacity and pulse effects
- Content complexity affects glitch intensity
- Color variance impacts chromatic aberration strength
- Image detail level influences scanline density
- Contrast levels affect noise distribution

## Contributing

Feel free to submit issues and pull requests to help improve the bot.

## License

MIT License

## Credits

Created by Graceus777, richinseattle for tetsuo-ai
Inspired by Katsuhiro Otomo's AKIRA
