import sqlite3
import json
import sys
import os
import numpy as np
import librosa
from search_clips import find_best_clips, reset_movie_usage

FMIN, FMAX = 65.0, 2093.0

def choose_fft_and_hop(sr: int, n_samples: int, win_ms: float = 25.0,
                       min_n_fft: int = 128, max_n_fft: int = 2048,
                       hop_div: int = 4):
    """
    Returns (n_fft, hop_length, win_length)
    - n_fft targets win_ms, clamped to signal length and rounded to pow2
    - hop_length = n_fft / hop_div (usually 4)
    """
    if n_samples <= 0:
        return min_n_fft, max(16, min_n_fft // hop_div), min_n_fft

    target = int(sr * win_ms / 1000.0)
    n_fft = min(target, n_samples, max_n_fft)

    # round down to power of 2
    n_fft = 2 ** int(np.floor(np.log2(max(n_fft, 2))))
    n_fft = max(min_n_fft, n_fft)

    hop_length = max(16, n_fft // hop_div)
    win_length = n_fft
    return n_fft, hop_length, win_length

def choose_n_mels(n_fft: int) -> int:
    if n_fft <= 256:
        return 32
    if n_fft <= 512:
        return 64
    if n_fft <= 1024:
        return 80
    return 128

def compute_audio_features(audio_segment, sr=16000):
    """Compute audio features from a numpy array. Used by both song matching and DB population."""
    if len(audio_segment) == 0:
        return None
    _, hop_length, _ = choose_fft_and_hop(sr, len(audio_segment))
    frame_length = min(2048, len(audio_segment))
    min_frame = int(np.ceil(sr / FMIN))
    if frame_length >= min_frame:
        f0, voiced_flag, _ = librosa.pyin(y=audio_segment, fmin=FMIN, fmax=FMAX, sr=sr, 
                                          hop_length=hop_length, frame_length=frame_length)
        pitch_mean = float(np.nanmedian(f0[voiced_flag])) if voiced_flag.any() else float('nan')
    else :
        pitch_mean = float('nan')
    energy = float(np.mean(librosa.feature.rms(y=audio_segment, hop_length=hop_length, frame_length=frame_length)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio_segment, hop_length=hop_length, frame_length=frame_length)))
    return {'pitch_mean': pitch_mean, 'energy': energy, 'zero_crossing_rate': zcr}


def extract_word_audio_features(audio_file, start, end, sr=16000):
    """Extract audio features for a specific word from an audio file."""
    duration = end - start
    if duration <= 0:
        return None
    try:
        audio_segment, _ = librosa.load(audio_file, sr=sr, offset=start, duration=duration, mono=True)
        features = compute_audio_features(audio_segment, sr)
        return {**features, 'duration': duration}
    except Exception as e:
        print(f"  Warning: Could not extract features: {e}")
        return None


def pair_lyrics_to_clips(lyrics_path, db_path='movie_clips.db', output_dir='./output', top_k=5):
    """
    For each word in the song lyrics, find the best matching movie clips.

    Args:
        lyrics_path: Path to lyrics.json (from create_song_subtitles.py)
        db_path: Path to movie_clips.db
        output_dir: Directory to save output
        top_k: Number of candidate clips per word

    Returns:
        List of matched clips for each word
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load lyrics
    with open(lyrics_path, 'r') as f:
        lyrics = json.load(f)

    vocals_file = lyrics['audio_file']
    words = lyrics['words']

    print(f"Matching {len(words)} words to movie clips...")

    # Connect to database
    db = sqlite3.connect(db_path)

    # Reset movie usage for fresh scoring
    reset_movie_usage()

    # Compute song average energy for relative loudness
    all_energies = []
    for word_data in words:
        f = extract_word_audio_features(vocals_file, word_data.get('start', 0), word_data.get('end', 0))
        if f and f.get('energy'):
            all_energies.append(f['energy'])
    song_avg_energy = float(np.mean(all_energies)) if all_energies else None

    # Match each word
    matched_clips = []
    for i, word_data in enumerate(words):
        word = word_data['text'].strip().lower()
        start = word_data.get('start', 0)
        end = word_data.get('end', 0)

        # Extract audio features from the song's word
        audio_features = extract_word_audio_features(vocals_file, start, end)
        if audio_features is None:
            continue

        word_energy = audio_features.get('energy')
        if word_energy and song_avg_energy:
            relative_gain_db = float(20 * np.log10(word_energy / song_avg_energy))
        else:
            relative_gain_db = 0.0

        # Find best clips
        results = find_best_clips(word, audio_features, db, top_k=top_k)

        # Build match entry
        match = {
            'word': word_data['text'],
            'song_start': start,
            'song_end': end,
            'song_duration': end - start,
            'candidates': []
        }

        for score, clip in results:
            match['candidates'].append({
                'score': round(score, 2),
                'movie_name': clip['movie_name'],
                'movie_path': clip['movie_path'],
                'movie_word': clip['word'],
                'timestamp': clip['timestamp'],
                'duration': clip['duration'],
                'end_timestamp': clip['end_timestamp'],
                'gain_db': relative_gain_db,
                'gain_pitch' : clip['gain_pitch'],
                'audio_stream_index': clip['audio_stream_index']
            })

        # Best match (auto-selected)
        if match['candidates']:
            match['selected'] = match['candidates'][0]
        else:
            match['selected'] = None
            print(f"  Warning: No match found for '{word}'")

        matched_clips.append(match)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(words)} words...")

    db.close()

    # Save results
    output = {
        'song_file': lyrics.get('audio_file', ''),
        'total_words': len(words),
        'matched_words': sum(1 for m in matched_clips if m['selected']),
        'matches': matched_clips
    }

    output_path = os.path.join(output_dir, 'matched_clips.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nMatching complete:")
    print(f"  Total words: {len(words)}")
    print(f"  Matched: {output['matched_words']}")
    print(f"  Saved to {output_path}")

    return output


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python pair_movie_to_song.py <lyrics.json> [db_path] [output_dir]")
        sys.exit(1)

    lyrics_path = sys.argv[1]
    db_path = sys.argv[2] if len(sys.argv) > 2 else 'movie_clips.db'
    output_dir = sys.argv[3] if len(sys.argv) > 3 else './output'
    pair_lyrics_to_clips(lyrics_path, db_path, output_dir)
