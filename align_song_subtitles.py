import sys
import re
import os
import json

import torch
import stable_whisper

def has_c_tags(vtt_path):
    with open(vtt_path, "r") as f:
        content = f.read()
    return bool(re.search(r'<\d{2}:\d{2}:\d{2}', content))

def extract_text_from_c_tags(vtt_path):
    with open(vtt_path, "r") as f:
        lines = f.readlines()
    clean_words = []
    for line in lines:
        line = line.strip()
        if not line or '-->' in line:
            continue
        if line.startswith(('WEBVTT', 'Kind:', 'Language:')):
            continue
        # Only keep lines that have timing tags — these are the "new content" lines
        # Plain repetition lines (no tags) are the rolling window carry-overs
        if '<' not in line and re.search(r'[a-zA-Z]', line):
            # plain text line — skip if it looks like a repetition (no new timing info)
            continue
        line = re.sub(r'<[^>]+>', '', line)
        line = re.sub(r'[♪♫]', '', line)
        line = re.sub(r'[^\w\s]', '', line)
        line = re.sub(r'[^a-zA-Z\s]', '', line)
        line = re.sub(r'\s+', ' ', line).strip()
        if line:
            clean_words.append(line)
    return " ".join(clean_words)


def extract_text_no_c_tags(vtt_path):
    """For simple VTTs without rolling window: just clean lines independently."""
    with open(vtt_path, "r") as f:
        lines = f.readlines()
    clean_words = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(('WEBVTT', 'Kind:', 'Language:')):
            continue
        if '-->' in line:
            continue
        line = re.sub(r'[♪♫]', '', line)
        line = re.sub(r'<[^>]+>', '', line)
        line = re.sub(r'[^\w\s]', '', line)
        line = re.sub(r'[^a-zA-Z\s]', '', line)
        line = re.sub(r'\s+', ' ', line).strip()
        if line:
            clean_words.append(line)
    return " ".join(clean_words)

def preprocessing_yt_subtitle(vtt_path):
    if has_c_tags(vtt_path):
        return extract_text_from_c_tags(vtt_path)
    return extract_text_no_c_tags(vtt_path)


def align_audio_subtitles(audio_path, text_path, output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = stable_whisper.load_model('small.en', device=device)

    text = preprocessing_yt_subtitle(text_path)

    result = model.align(audio_path, text, 'English')
    try:
        model.refine(audio_path, result)
    except RuntimeError as e:
        print(f"  Warning: refine skipped ({e})")

    result_dict = result.to_dict()
    word_timestamps = []
    for seg in result_dict['segments']:
        for word in seg['words']:
            word_timestamps.append({
                'text': word['word'].strip(),
                'start': word['start'],
                'end': word['end'],
                'score': word['probability']
            })

    metadata = {'audio_file': audio_path, 'words': word_timestamps}
    metadata_path = os.path.join(output_dir, 'alignment.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Extraction complete. Metadata saved to {metadata_path}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python align_song_subtitles.py <song_vocal> <song_subtitle> [output_dir]")
        sys.exit(1)

    song_vocal = sys.argv[1]
    song_subtitle = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else './output'
    align_audio_subtitles(song_vocal, song_subtitle, output_dir)