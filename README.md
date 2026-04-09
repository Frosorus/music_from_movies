## Music from Movies
Takes a YouTube song and assembles a video from matching movie clips.

## Pipeline
1. Download song + subtitles (yt-dlp)
2. Separate vocals (Demucs)
3. Align lyrics word-level (stable-ts)
4. Match words to movie clips (librosa features)
5. Assemble final video (MoviePy/ffmpeg)

## Usage
./pipeline.sh <youtube_url> <output_dir>
