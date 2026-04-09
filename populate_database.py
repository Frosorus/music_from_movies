#!/usr/bin/env python3
"""
Full database rebuild pipeline.
Drops and recreates the database, then processes all movies in audio_subtitles.json.
Usage: python populate_database.py [db_path] [audio_subtitles.json]
"""

import json, sqlite3, subprocess, gc, html, re, string, warnings
import numpy as np
import torch
from pathlib import Path
from metaphone import doublemetaphone
from tqdm import tqdm
import pysrt
import stable_whisper
import time

from pair_movie_to_song import compute_audio_features

warnings.filterwarnings('ignore')

EXTRACTED_DIR = Path("/mnt/e/extracted_english/")
SR = 16000



# ─── DATABASE ────────────────────────────────────────────────────────────────

def create_database(db_path):
    db = sqlite3.connect(db_path)
    db.execute('''
        CREATE TABLE IF NOT EXISTS clips (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            movie_name         TEXT,
            movie_path         TEXT,
            audio_stream_index INTEGER,
            word               TEXT,
            phonetic           TEXT,
            timestamp          REAL,
            duration           REAL,
            end_timestamp      REAL,
            pitch_mean         REAL,
            energy             REAL,
            zero_crossing_rate REAL,
            context            TEXT,
            confidence_score   REAL
        )
    ''')
    db.execute('CREATE INDEX IF NOT EXISTS idx_movie             ON clips(movie_name)')
    db.execute('CREATE INDEX IF NOT EXISTS idx_word_dur_conf     ON clips(word, duration, confidence_score)')
    db.execute('CREATE INDEX IF NOT EXISTS idx_phonetic_dur_conf ON clips(phonetic, duration, confidence_score)')
    db.execute('CREATE INDEX IF NOT EXISTS idx_dur_conf          ON clips(duration, confidence_score)')
    db.commit()
    return db


# ─── AUDIO ───────────────────────────────────────────────────────────────────

def extract_audio_segment(video_path, audio_index, start, duration, sr=SR):
    """Decode audio directly from video via ffmpeg pipe — no temp files stored."""
    cmd = [
        'ffmpeg', '-v', 'error',
        '-ss', str(max(0.0, start)),
        '-i', str(video_path),
        '-map', f'0:{audio_index}',
        '-t', str(duration),
        '-ac', '1', '-ar', str(sr),
        '-f', 'f32le', 'pipe:1'
    ]
    result = subprocess.run(cmd, capture_output=True)
    audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
    # Trim or pad to exact expected length
    expected = int(duration * sr)
    if len(audio) < expected:
        audio = np.pad(audio, (0, expected - len(audio)))
    return audio[:expected]



# ─── SUBTITLE ────────────────────────────────────────────────────────────────

def clean_sub_text(text):
    text = html.unescape(text)
    text = re.sub(r"</?[^>]+>", "", text)
    text = re.sub(r"\{\\.*?\}", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\[.*?\]", "", text)   # remove [stage directions]
    text = re.sub(r"\(.*?\)", "", text)   # remove (sound effects)
    text = text.translate(str.maketrans('', '', string.punctuation + "«»\u201c\u201d\u2018\u2019"))
    return re.sub(r"\s+", " ", text).strip().lower()


def extract_srt(video_path, subtitle_index, srt_path):
    """Extract SRT from video to disk (kept permanently — small file)."""
    if srt_path.exists():
        return True
    r = subprocess.run(
        ['ffmpeg', '-v', 'error', '-i', str(video_path),
         '-map', f'0:{subtitle_index}', str(srt_path)],
        capture_output=True
    )
    return r.returncode == 0


# ─── ALIGNMENT ───────────────────────────────────────────────────────────────

def load_aligner(device):
    return stable_whisper.load_model('small.en', device=device)  # or "small", "medium"

# ─── MOVIE PROCESSING ────────────────────────────────────────────────────────

def process_movie(movie_path, audio_index, subtitle_index, db, model):
    movie_name = Path(movie_path).with_suffix("").name
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    srt_path = EXTRACTED_DIR / f"{movie_name}.subs.srt"

    if not extract_srt(movie_path, subtitle_index, srt_path):
        print(f"  ✗ Could not extract subtitles")
        return 0

    try:
        subs = pysrt.open(str(srt_path))
    except Exception as e:
        print(f"  ✗ Could not read SRT: {e}")
        return 0

    inserted = 0
    torch.cuda.empty_cache()
    result = []
    for sub in tqdm(subs, desc=f"  {movie_name[:40]}", leave=False):
        text = clean_sub_text(sub.text)
        if not text:
            continue

        seg_start = (sub.start.hours * 3600 + sub.start.minutes * 60
                     + sub.start.seconds + sub.start.milliseconds / 1000)
        seg_end = (sub.end.hours * 3600 + sub.end.minutes * 60
                   + sub.end.seconds + sub.end.milliseconds / 1000)
        result.append({'start':seg_start, "end":seg_end, 'text':text})

    merged = []
    for seg in result:
        if merged and seg['start'] < merged[-1]['end']:
            # overlap — extend the previous segment
            merged[-1]['end'] = max(merged[-1]['end'], seg['end'])
            merged[-1]['text'] += ' ' + seg['text']
        else:
            merged.append(seg)
    result = model.align_words(movie_path, merged, 'English')
    try:
        model.refine(movie_path, result)
    except RuntimeError as e:
        print(f"  Warning: refine skipped ({e})")
    
    print("Loading audio from movie")
    time_audio = time.time()
    cmd = ['ffmpeg', '-v', 'error', '-i', str(movie_path),
       '-map', f'0:{audio_index}', '-ac', '1', '-ar', str(SR), '-f', 'f32le', 'pipe:1']
    raw = subprocess.run(cmd, capture_output=True).stdout
    full_audio = np.frombuffer(raw, dtype=np.float32).copy()
    print("\t{} s".format(time.time()-time_audio))
    result = result.to_dict()
    print("Adding to database")
    for seg in tqdm(result['segments']):
        for word in seg['words']:
            text = word['word'].strip().lower()
            if not text:
                continue

            abs_start = word['start']
            abs_end = word['end']
            duration = abs_end - abs_start
            s = int(abs_start * SR)
            e = int(abs_end * SR)
            word_audio = full_audio[s:e]

            features = compute_audio_features(word_audio, SR)
            if features is None:
                continue

            phonetic = doublemetaphone(text)[0] or text.upper()

            db.execute('''
                INSERT INTO clips
                (movie_name, movie_path, audio_stream_index, word, phonetic,
                    timestamp, duration, end_timestamp,
                    pitch_mean, energy, zero_crossing_rate,
                    context, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                movie_name, str(movie_path), audio_index,
                text, phonetic,
                abs_start, duration, abs_end,
                features['pitch_mean'], features['energy'], features['zero_crossing_rate'],
                text,  # subtitle line as context
                word['probability']
            ))
            inserted += 1

            del word_audio

    db.commit()
    return inserted


# ─── PIPELINE ────────────────────────────────────────────────────────────────

def run_pipeline(json_path, db, skip_movies=None):
    """Process all movies in json_path, skipping any in skip_movies."""
    with open(json_path) as f:
        movie_data = json.load(f)

    skip_movies = skip_movies or set()

    to_process = [
        (p, info) for p, info in movie_data.items()
        if info.get('audio_stream_index') is not None
        and info.get('subtitle_stream_index') is not None
        and Path(p).with_suffix("").name not in skip_movies
    ]

    if not to_process:
        print("Nothing to process.")
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading alignment model on {device}...")
    model = load_aligner(device)

    total = 0
    for i, (movie_path, info) in enumerate(to_process, 1):
        movie_name = Path(movie_path).with_suffix("").name
        print(f"\n[{i}/{len(to_process)}] {movie_name}")
        n = process_movie(
            movie_path, info['audio_stream_index'], info['subtitle_stream_index'],
            db, model
        )
        total += n
        print(f"  ✓ {n} clips inserted")
        gc.collect()

    return total


if __name__ == '__main__':
    import sys
    db_path = sys.argv[1] if len(sys.argv) > 1 else 'movie_clips.db'
    json_path = sys.argv[2] if len(sys.argv) > 2 else 'audio_subtitles.json'

    if Path(db_path).exists():
        Path(db_path).unlink()
        print(f"Dropped existing database: {db_path}")

    db = create_database(db_path)
    total = run_pipeline(json_path, db)
    db.close()
    print(f"\nTotal clips inserted: {total:,}")
