import json
import sys
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import shutil
from tqdm import tqdm
import random



def extract_clip_ffmpeg(video_path, start, word_duration_song, total_duration, output_path, ratio, audio_path, gain_db, gain_pitch, audio_stream_index):
    """
    Extract a video clip using ffmpeg.
    - Video runs for total_duration (extended to fill gap to next word)
    - Audio only plays during word_duration, then silence for the rest

    Args:
        video_path: Path to movie file
        start: Start timestamp in movie
        word_duration_song: How long the actual word lasts (audio plays here)
        word_duration_movie: How long it takes in the movie to be said
        total_duration: Extended duration to fill gap to next word (video plays here)
        output_path: Where to save the clip
    """
    TARGET_FPS = 30
    
    video_filter = (f"[0:v]setpts=(PTS-STARTPTS)/{ratio},trim=0:{total_duration},setpts=PTS-STARTPTS,"
        f'scale=1920:1080:force_original_aspect_ratio=decrease,'
        f'pad=1920:1080:(ow-iw)/2:(oh-ih)/2,'
        f'fps={TARGET_FPS},format=yuv420p[vout]')

    # Audio filter: keep audio for word_duration, then pad with silence for the rest
    pitch_ratio = 2 ** (gain_pitch / 12)
    if total_duration > 0.1 and abs(gain_pitch) > 0.5:
        pitch_filter = f'rubberband=pitch={pitch_ratio:.6f},'
    else:
        pitch_filter = ''
    # apad pads with silence, atrim limits total audio length
    audio_filter = (
        f'[1:{audio_stream_index}]atempo={ratio},{pitch_filter}loudnorm=I=-23:LRA=7:TP=-2,volume={gain_db}dB,'
        f'asetpts=PTS-STARTPTS,atrim=start=0:end={word_duration_song},'
        f'apad,atrim=0:{total_duration},'
        f'aformat=sample_rates=48000:channel_layouts=stereo[aout]'
    )

    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start),
        '-i', video_path,
        '-ss', str(start),
        '-i', audio_path,
        '-t', str(total_duration),
        "-filter_complex", f"{video_filter};{audio_filter}",
        '-map', '[vout]',
        '-map', '[aout]',

        '-c:v', 'h264_nvenc',
        '-preset', 'p1',
        '-rc', 'vbr',
        '-cq:v', '22',
        '-color_primaries', 'bt709',
        '-color_trc', 'bt709',
        '-colorspace', 'bt709',
        '-b:v', '0',
        
        '-c:a', 'aac', 
        '-b:a', '160k',
        '-ar', '48000',
        '-ac', "2",
        
        "-fflags", "+genpts",
        '-avoid_negative_ts', 'make_zero',
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        print(result.stderr)
        print(result.stdout)
    return result.returncode == 0


def is_valid_clip(path):
    """Return True if ffprobe can read a video stream from the clip."""
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=codec_type',
         '-of', 'default=noprint_wrappers=1:nokey=1', path],
        capture_output=True, text=True
    )
    return r.returncode == 0 and r.stdout.strip() != ''


def generate_black_screen(duration, output_path):
    """Generate a black 1920x1080 clip with silence for the given duration."""
    duration = max(duration, 0.1)
    cmd = [
        'ffmpeg', '-y',
        '-f', 'lavfi', '-i', 'color=c=black:size=1920x1080:rate=30',
        '-f', 'lavfi', '-i', 'anullsrc=channel_layout=stereo:sample_rate=48000',
        '-t', str(duration),
        '-c:v', 'h264_nvenc', '-preset', 'p1', '-rc', 'vbr', '-cq:v', '22',
        '-pix_fmt', 'yuv420p',
        '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
        '-c:a', 'aac', '-b:a', '160k', '-ar', '48000', '-ac', '2',
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def build_instrumental_ffmpeg(metadata_path, output_dir):
    """
    Mix all non-vocal stems into one instrumental track using ffmpeg.
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    stems = metadata['stems']
    instrumental_stems = []
    for stem_name, stem_path in stems.items():
        if stem_name == 'vocals':
            continue
        print(f"  Stem: {stem_name} -> {stem_path}")
        instrumental_stems.append(stem_path)

    if not instrumental_stems:
        raise ValueError("No instrumental stems found")

    instrumental_path = os.path.join(output_dir, 'instrumental.wav')

    cmd = ['ffmpeg', '-y']
    for stem in instrumental_stems:
        cmd.extend(['-i', stem])
    cmd.extend([
        '-filter_complex', f'amix=inputs={len(instrumental_stems)}:duration=longest',
        instrumental_path
    ])

    print(f"  Mixing {len(instrumental_stems)} stems...")
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(f"  Instrumental saved to {instrumental_path}")
    return instrumental_path


def assemble_music_video(matched_clips_path, metadata_path, output_dir='./output', output_filename='final_video.mp4'):
    """
    Assemble the final music video using pure ffmpeg (no moviepy):
    1. Build instrumental track
    2. Extract each clip with ffmpeg -> temp files
    3. Concatenate clips with ffmpeg concat demuxer
    4. Add instrumental audio
    """
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix='music_video_')
    
    print(f"Temp directory: {temp_dir}")

    try:
        with open(matched_clips_path, 'r') as f:
            matched_data = json.load(f)

        matches = matched_data['matches']

        # 1. Build instrumental track
        print("Building instrumental track...")
        instrumental_path = build_instrumental_ffmpeg(metadata_path, output_dir)
        
        # 3. Filter valid matches and compute extended durations
        # actual_video_path is None when a black screen should be used instead
        valid_matches = []
        for match in matches:
            selected = match.get('selected')
            if selected is None:
                valid_matches.append((match, None))
                continue
            audio_path = selected.get('movie_path')
            actual_video_path = audio_path
            if not audio_path or not os.path.exists(actual_video_path):
                valid_matches.append((match, None))
                continue
            valid_matches.append((match, actual_video_path))
        
        # Sort by song timestamp
        valid_matches.sort(key=lambda x: x[0]['song_start'])

        # Get total song duration from instrumental
        probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                     '-of', 'default=noprint_wrappers=1:nokey=1', instrumental_path]
        song_duration = float(subprocess.run(probe_cmd, capture_output=True, text=True).stdout.strip())

        # Compute extended durations: each clip extends until the next word starts
        print(f"\nExtracting {len(valid_matches)} video clips...")
        clip_files = []
        
        clip_path = os.path.join(temp_dir, f'clip_intro.mkv')
        total_duration = valid_matches[0][0]['song_start']
        real_matches = [(m, p) for m, p in valid_matches if p is not None]
        if real_matches:
            match, actual_video_path = real_matches[int(random.random() * len(real_matches))]
            selected = match['selected']
            success = extract_clip_ffmpeg(
                actual_video_path,
                selected['timestamp'],
                0,
                total_duration,
                clip_path,
                1,
                selected['movie_path'],
                1,
                0.0,
                selected['audio_stream_index']
            )
            if not success or not os.path.exists(clip_path) or os.path.getsize(clip_path) == 0:
                success = generate_black_screen(total_duration, clip_path)
        else:
            success = generate_black_screen(total_duration, clip_path)
        clip_files.append({
            'path': clip_path,
            'song_start': 0,
            'duration': total_duration
        })
        for idx, (match, actual_video_path) in enumerate(tqdm(valid_matches, desc="Extracting clips")):
            selected = match['selected']
            song_start = match['song_start']
            word_duration = match['song_duration']

            # Extend video to fill gap until next word (or song end)
            if idx < len(valid_matches) - 1:
                next_song_start = valid_matches[idx + 1][0]['song_start']
                total_duration = next_song_start - song_start
            else:
                total_duration = song_duration - song_start

            total_duration = max(word_duration, total_duration)

            # nvenc produces corrupt MKV for very short durations — enforce a minimum
            total_duration = max(total_duration, 0.1)

            clip_path = os.path.join(temp_dir, f'clip_{idx:04d}.mkv')
            if actual_video_path is None:
                success = generate_black_screen(total_duration, clip_path)
            else:
                movie_timestamp = selected['timestamp']
                word_duration_movie = selected['duration']
                audio_path = selected['movie_path']
                ratio = word_duration_movie / word_duration
                gain_db = selected['gain_db']
                gain_pitch = selected['gain_pitch']
                audio_stream_index = selected['audio_stream_index']
                success = extract_clip_ffmpeg(
                    actual_video_path,
                    movie_timestamp,
                    word_duration,
                    total_duration,
                    clip_path,
                    ratio,
                    audio_path,
                    gain_db,
                    gain_pitch,
                    audio_stream_index
                )
                if not success or not os.path.exists(clip_path) or os.path.getsize(clip_path) == 0:
                    print(f"  Warning: clip export failed for '{match['word']}', using black screen")
                    success = generate_black_screen(total_duration, clip_path)

            # Validate the clip regardless of how it was produced — nvenc can
            # return exit code 0 but write a corrupt MKV in edge cases
            if os.path.exists(clip_path) and not is_valid_clip(clip_path):
                print(f"  Warning: corrupt clip detected for '{match.get('word', '?')}', using black screen")
                success = generate_black_screen(total_duration, clip_path)

            if success and os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                clip_files.append({
                    'path': clip_path,
                    'song_start': song_start,
                    'duration': total_duration
                })

        print(f"\nExtracted {len(clip_files)} clips successfully")

        if not clip_files:
            print("Error: No video clips could be extracted")
            return None

        # Sort clips by song timestamp
        clip_files.sort(key=lambda x: x['song_start'])

        # 4. Create ffmpeg concat file
        concat_file = os.path.join(temp_dir, 'concat.txt')
        with open(concat_file, 'w') as f:
            for clip_info in clip_files:
                # Escape single quotes in path for ffmpeg
                escaped_path = clip_info['path'].replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")
                f.write(f"duration {clip_info['duration']:.6f}\n")

        # 5. Concatenate all clips (no re-encoding, very fast)
        video_only_path = os.path.join(temp_dir, 'video_only.mp4')
        print("\nConcatenating clips...")
        concat_cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',
            video_only_path
        ]
        result = subprocess.run(concat_cmd, capture_output=True, text=True)

        concat_failed = result.returncode != 0 or 'Impossible to open' in result.stderr or 'invalid' in result.stderr
        if concat_failed:
            print("  Concat copy failed, re-encoding...")
            concat_cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '18',
                '-pix_fmt', 'yuv420p',
                video_only_path
            ]
            subprocess.run(concat_cmd, capture_output=True, text=True, check=True)

        # 6. Mix movie dialogue audio + instrumental, combine with video
        output_path = os.path.join(output_dir, output_filename)
        print(f"\nMixing movie audio + instrumental and rendering to {output_path}...")

        final_cmd = [
            'ffmpeg', '-y',
            '-i', video_only_path,       # Video + movie dialogue audio
            '-i', instrumental_path,     # Instrumental track
            '-filter_complex',
            '[0:a][1:a]amix=inputs=2:duration=shortest:weights=1 1[aout]',
            '-map', '0:v:0',            # Video from concat
            '-map', '[aout]',            # Mixed audio
            '-c:v', 'copy',
            '-c:a', 'aac', '-b:a', '192k',
            output_path
        ]
        subprocess.run(final_cmd, capture_output=True, text=True, check=True)

        print(f"\nDone! Final video saved to {output_path}")
        return output_path

    finally:
        print(f"\nCleaning up temp files...")
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python assemble_video.py <matched_clips.json> <extract_metadata.json> [output_dir]")
        sys.exit(1)

    matched_clips_path = sys.argv[1]
    metadata_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else './output'
    assemble_music_video(matched_clips_path, metadata_path, output_dir)
