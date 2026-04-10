import sqlite3
import json
import re
import numpy as np
from metaphone import doublemetaphone
from scipy.spatial.distance import cosine
from collections import Counter


def clean_word(word):
    """Strip everything except letters"""
    word = re.sub(r"[^a-zA-Z]", "", word)
    return word.lower()


# Track movie usage PER WORD for diversity scoring
# Key: (word, movie_name) -> count
# Penalizes using same movie for same word, but allows reusing movies for different words
word_movie_usage = Counter()

# Column names matching the database schema
COLUMNS = [
    'id','movie_name', 'movie_path', 'audio_stream_index', 'word', 'phonetic',
    'timestamp', 'duration', 'end_timestamp',
    'pitch_mean', 'energy', 'zero_crossing_rate',
    'context', 'confidence_score'
]



def row_to_dict(row):
    """Convert a database row to a dictionary"""
    return dict(zip(COLUMNS, row))

def duration_penalty(duration_music, duration_movie, k = 0.15, p=3):
    """
    Returns a score multiplier based on the ratio of the duration of the move
    - 1 when same value
    - drops gently for value not too far apart
    - drops abruptly for bigger mismatch
    
    :param duration_music: duration of the word in the lyrics
    :param duration_movie: duration of the word in the movie
    :param k: controls the knee, where it starts dropping
    :param p: controls how sharply it falls after the knee
    """

    r = min(duration_movie, duration_music)/max(duration_movie, duration_music)
    m = 1.0 - r
    return 1.0 /(1.0 + (m/k)**p)

def pitch_penalty(pitch_music, pitch_movie):
    music_valid = pitch_music and not np.isnan(pitch_music)
    movie_valid = pitch_movie and not np.isnan(pitch_movie)
    if music_valid and movie_valid:
        n_steps = 12 * np.log2(pitch_music / pitch_movie)
        return max(0, 1 - abs(n_steps) / 12)
    if music_valid and not movie_valid:
        return 0.5  # song has pitch target, clip can't be shifted
    return 1  # no target, neutral


def score_clip(lyric_word, lyric_audio_features, clip, lyric_phonetic):
    """
    Score a candidate clip based on:
    - Exact word match (highest priority)
    - Phonetic match
    - Audio similarity (pitch, energy, zcr) - only on filtered candidates
    - Movie diversity (penalty for overused movies)
    """
    score = 0.0

    # 1. EXACT MATCH bonus
    if clip['word'].lower() == lyric_word.lower():
        score += 100

    # 2. PHONETIC MATCH bonus
    if clip['phonetic'] == lyric_phonetic:
        score += 50

    # 3. AUDIO SIMILARITY (ranking among filtered candidates)
    # 3a. Duration similarity - multiplicative penalty
    dur_score = duration_penalty(lyric_audio_features['duration'], clip['duration'])
    # 3b. Pitch similarity - multiplicative penalty
    pitch_score = pitch_penalty(lyric_audio_features['pitch_mean'], clip['pitch_mean'])
    # 3c. Energy similarity (loudness) - 0 to 10 points
    try:
        energy_diff = abs(clip['energy'] - lyric_audio_features['energy'])
        energy_similarity = max(0, 1 - (energy_diff / 0.1))
        score += energy_similarity * 10
    except (TypeError, KeyError):
        pass
    #3d. ZCR 
    zcr_diff = abs(clip['zero_crossing_rate'] - lyric_audio_features['zero_crossing_rate'])
    zcr_similarity = max(0, 1 - (zcr_diff / 0.2))
    score += zcr_similarity * 10
    # 4. MOVIE DIVERSITY penalty (-1 per previous use of same movie for same word, Make sure highest quality sound with tiny penalty)
    usage_penalty = word_movie_usage[(lyric_word.lower(), clip['movie_name'])] * 1
    score -= usage_penalty

    return dur_score*pitch_score*score

def gain_db_from_rms(rms_ref, rms_src, eps=1e-12):
    rms_ref = max(rms_ref, eps)
    rms_src = max(rms_src, eps)
    return float(20.0 * np.log10(rms_ref / rms_src))

def gain_pitch(pitch_song, pitch_movie):
    if pitch_song and pitch_movie and not np.isnan(pitch_song) and not np.isnan(pitch_movie):
        return float(12 * np.log2(pitch_song / pitch_movie))
    else:
        return 0.0

def find_best_clips(lyric_word, lyric_audio_features, db, top_k=5):
    """
    Efficient two-stage search:
    Stage 1: Fast filtering by exact word / phonetic match (indexed)
    Stage 2: Rank filtered candidates by audio similarity
    Fallback: If no matches found, search by audio similarity on a sample
    """
    # Clean the word: remove punctuation, normalize
    cleaned_word = clean_word(lyric_word)
    if not cleaned_word:
        return []

    lyric_phonetic = doublemetaphone(cleaned_word)[0]
    
    dur = float(lyric_audio_features["duration"])
    dmin, dmax = 0.5*dur, 2.0*dur
    conf_min = 0.25
    # STAGE 1: Fast filtering by word/phonetic (uses database indexes)
    rows = db.execute("""
        SELECT *
        FROM clips
        WHERE (word = ? OR (phonetic = ? AND word != ?))
          AND duration BETWEEN ? AND ?
          AND confidence_score >= ?
        LIMIT 800
    """, (cleaned_word, lyric_phonetic, cleaned_word, dmin, dmax, conf_min)).fetchall()
    if not rows:
        # cheaper fallback than ORDER BY RANDOM()
        print("No exact word or phonetic match")
        rows = db.execute("""
            SELECT *
            FROM clips
            WHERE duration BETWEEN ? AND ?
              AND confidence_score >= ?
            LIMIT 1000
        """, (dmin, dmax, conf_min)).fetchall()
        if not rows:
            print("No word with good enough confidence score")
            rows = db.execute("""
            SELECT *
            FROM clips
            WHERE duration BETWEEN ? AND ?
            LIMIT 1000
        """, (dmin, dmax)).fetchall()
    
    candidates = [row_to_dict(r) for r in rows]

    for clip in candidates:
        # print(f"energy ref: {lyric_audio_features['energy']}, clip energy: {clip['energy']}, clip word: {clip.get('word')}, movie: {clip.get('movie_name')}")

        clip["gain_db"] = gain_db_from_rms(lyric_audio_features["energy"], clip["energy"])
        clip["gain_pitch"] = gain_pitch(lyric_audio_features.get('pitch_mean'), clip['pitch_mean']) 
    # STAGE 2: Score candidates
    scored_clips = [(score_clip(cleaned_word, lyric_audio_features, clip, lyric_phonetic), clip)
              for clip in candidates]

    # Sort by score descending
    scored_clips.sort(key=lambda x: x[0], reverse=True)

    # Update usage: track (word, movie) pair
    if scored_clips:
        selected_movie = scored_clips[0][1]['movie_name']
        word_movie_usage[(lyric_word.lower(), selected_movie)] += 1

    return scored_clips[:top_k]


def reset_movie_usage():
    """Reset movie usage counter (call before generating a new video)"""
    word_movie_usage.clear()


# ---- Example usage ----
if __name__ == '__main__':
    db = sqlite3.connect('movie_clips.db')

    # Example: search for the word "fire"
    # In real usage, lyric_audio_features would come from your song's audio
    example_audio_features = {
        'mfcc_mean': [0] * 13,
        'pitch_mean': 200.0,
        'energy': 0.05,
        'zero_crossing_rate': 0.1
    }

    results = find_best_clips("fire", example_audio_features, db, top_k=5)

    print(f"\nTop 5 matches for 'fire':")
    print("-" * 60)
    for score, clip in results:
        print(f"  Score: {score:.2f}")
        print(f"  Movie: {clip['movie_name']}")
        print(f"  Word:  {clip['word']}")
        print(f"  Time:  {clip['timestamp']:.2f}s - {clip['end_timestamp']:.2f}s")
        print(f"  Context: {clip['context']}")
        print()

    db.close()
