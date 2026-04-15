#!/bin/bash
# pipeline.sh - Orchestrates the full music video pipeline
# Usage: ./pipeline.sh <song_file> <output_dir>

set -e  # Stop on any error

SONG_LINK="$1"
OUTPUT_DIR="$2"

# Path to the bin/ directory of your main conda env (yt-dlp, demucs, moviepy, librosa…)
# Override via environment variable or edit here, e.g.:
#   EXTRACT_SONG_BIN="/path/to/miniconda3/envs/extract_song/bin/"
EXTRACT_SONG_PY="${EXTRACT_SONG_BIN:-}"

# Full path to the Python executable in your stable-ts conda env
# Override via environment variable or edit here, e.g.:
#   STABLE_TS_PYTHON="/path/to/miniconda3/envs/stable-ts/bin/python"
STABLE_TS_PY="${STABLE_TS_PYTHON:-python}"

if [ -z "$SONG_LINK" ]; then
    echo "Usage: ./pipeline.sh <song_link> <output_dir>"
    exit 1
fi

echo "=========================================="
echo "  Music From Movies Pipeline"
echo "  Song: $SONG_LINK"
echo "  Output: $OUTPUT_DIR"
echo "=========================================="

echo ""
echo "=== Step 0: Download music and auto subtitles ==="
SONG_TITLE=$("${EXTRACT_SONG_PY}yt-dlp" --print "%(title)s" --skip-download "$SONG_LINK")
# Strip down the song title in order to be more suitable for naming
SONG_TITLE=$(echo "$SONG_TITLE" | tr ' ' '_' | tr -cd '[:alnum:]_')
"${EXTRACT_SONG_PY}yt-dlp" --write-auto-sub --remux-video mp4 -P "./data/${SONG_TITLE}/" -o "song.%(ext)s" -o "subtitle:subtitle" "$SONG_LINK"
if [ ! -f "./data/${SONG_TITLE}/subtitle.en.vtt" ]; then
    mv "./data/${SONG_TITLE}/subtitle.en"*.vtt "./data/${SONG_TITLE}/subtitle.en.vtt"
fi

echo ""
echo "=== Step 1: Extract vocals with Demucs ==="
"${EXTRACT_SONG_PY}python" extract_audio_from_song.py "./data/${SONG_TITLE}/song.mp4" "${OUTPUT_DIR}/${SONG_TITLE}"

echo ""
echo "=== Step 2: Align auto generated Youtube subtitles ==="
$STABLE_TS_PY align_song_subtitles.py "$OUTPUT_DIR/${SONG_TITLE}/vocals.wav" "./data/${SONG_TITLE}/subtitle.en.vtt" "${OUTPUT_DIR}/${SONG_TITLE}"

echo ""
echo "=== Step 3: Match lyrics to movie clips ==="
"${EXTRACT_SONG_PY}python" pair_movie_to_song.py "$OUTPUT_DIR/${SONG_TITLE}/alignment.json" "movie_clips.db" "$OUTPUT_DIR/${SONG_TITLE}"

echo ""
echo "=== Step 4: Assemble final video ==="
"${EXTRACT_SONG_PY}python" assemble_video.py "$OUTPUT_DIR/${SONG_TITLE}/matched_clips.json" "$OUTPUT_DIR/${SONG_TITLE}/extract_metadata.json" "$OUTPUT_DIR/${SONG_TITLE}"

echo ""
echo "=========================================="
echo "  Pipeline complete!"
echo "  Final video: $OUTPUT_DIR/${SONG_TITLE}/final_video.mp4"
echo "=========================================="
