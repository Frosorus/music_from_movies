import moviepy as mp
import numpy as np
import demucs.api
import torch
import sys
import os
import json


def load_video_silent(filepath):
    """Load video without ffmpeg spam"""
    stderr_backup = sys.stderr
    try:
        sys.stderr = open(os.devnull, 'w')
        video = mp.VideoFileClip(filepath)
    finally:
        sys.stderr.close()
        sys.stderr = stderr_backup
    return video


def extract_audio(song_file, output_dir='./output'):
    """
    Extract vocals and other stems from a song file using Demucs.

    Args:
        song_file: Path to the song (mp4, mp3, wav, etc.)
        output_dir: Directory to save output files

    Returns:
        dict with paths to the separated stems
    """
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading: {song_file}")
    video = load_video_silent(song_file)
    audio_array = video.audio.to_soundarray(fps=44100)
    audio_array = audio_array.T
    audio_array = audio_array.astype(np.float32)

    audio_tensor = torch.from_numpy(audio_array).float()

    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    print("Separating audio with Demucs...")
    separator = demucs.api.Separator()
    _, separated = separator.separate_tensor(audio_tensor)

    # Save each stem
    output_paths = {}
    for stem, source in separated.items():
        output_path = os.path.join(output_dir, f"{stem}.wav")
        demucs.api.save_audio(source, output_path, samplerate=separator.samplerate)
        output_paths[stem] = output_path
        print(f"  Saved {stem} -> {output_path}")

    # Save metadata for next step
    metadata = {
        'song_file': song_file,
        'stems': output_paths,
        'samplerate': separator.samplerate
    }
    metadata_path = os.path.join(output_dir, 'extract_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Extraction complete. Metadata saved to {metadata_path}")
    return metadata


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python extract_audio_from_song.py <song_file> [output_dir]")
        sys.exit(1)

    song_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else './output'
    extract_audio(song_file, output_dir)
