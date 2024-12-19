# Tetimi -- Tetsuo Discord Bot Commands

## Image Processing Commands

### !image
Process an image with various effects. Can be used with an attached image or will use the default input image.

**Basic Usage:**
```
!image [options]
!image --random  # Process a random image from images folder
```

**Effect Parameters:**
- `--rgb <r> <g> <b> --rgbalpha <0-255>` - Add RGB color overlay
- `--color <r> <g> <b> --coloralpha <0-255>` - Add color tint
- `--glitch <1-50>` - Apply glitch effect
- `--chroma <1-40>` - Add chromatic aberration
- `--scan <1-200>` - Add scan lines
- `--noise <0-1>` - Add noise effect
- `--energy <0-1>` - Add energy effect
- `--pulse <0-1>` - Add pulse effect

**Preset Effects:**
```
!image --preset <preset_name>
```
Available presets:
- cyberpunk
- vaporwave
- glitch_art
- retro
- matrix
- synthwave
- akira
- tetsuo
- neo_tokyo
- psychic
- tetsuo_rage

**Animation Options:**
```
!image --animate [options]
```
- `--frames <number>` - Number of frames (default: 30, max: 120)
- `--fps <number>` - Frames per second (default: 24, max: 60)
- `--style <style_name>` - Animation style

Available animation styles:
- glitch_surge
- power_surge
- psychic_blast
- digital_decay
- neo_flash

**Special Effects:**
- `--points` - Apply points effect
  - `--dot-size <size>` - Set dot size for points effect
  - `--reg-offset <offset>` - Set registration offset for points effect

### !testanimate
Test the animation system with a sequence of effect tests.
```
!testanimate
```

## Art Repository Commands

### !store
Store artwork in the repository.
```
!store <title> [tags...]
```
- Requires an attached image
- Tags are optional; if not provided, words from title are used as tags

### !process_store
Process an image with effects and store the result.
```
!process_store [effect_options]
```
- Requires an attached image
- Supports all effect parameters from !image command
- Automatically generates tags based on applied effects

### !remix
Apply effects to existing artwork from the repository.
```
!remix <artwork_id> [effect_options]
```
- Can optionally attach a new base image
- Supports all effect parameters from !image command

### !search
Search for artwork in the repository.
```
!search <query>
```
- Searches through titles, tags, and creator information
- Returns up to 50 results

### !trending
Show trending artwork from the last 24 hours.
```
!trending
```
- Displays most popular pieces based on interactions
- Shows creator and interaction count

### !history
Show modification history of specific artwork.
```
!history <artwork_id>
```
- Displays version history
- Shows modifiers and timestamps

## Utility Commands

### !help
Display help information about available commands.
```
!help
```
- Shows basic command usage
- Lists available effects and presets

## Image Management
- React with 🗑️ to delete bot-generated images
- All processed images are automatically saved with timestamp in filename

## Technical Notes
- Maximum file size limited by Discord (8MB for non-nitro servers)
- Animation frames limited to 120 frames
- FPS limited to 60
- All color values must be between 0-255
- Alpha values must be between 0-255
- Effect intensities have specific ranges as noted in parameters

