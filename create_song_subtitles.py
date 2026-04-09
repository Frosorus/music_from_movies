import os
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

import whisperx
import json
import sys


def transcribe_lyrics(audio_file, output_dir='./output', device='cuda', compute_type='float16', batch_size=16):
    """
    Transcribe lyrics from a vocals audio file using WhisperX.

    Args:
        audio_file: Path to vocals audio file (from Demucs)
        output_dir: Directory to save output
        device: 'cuda' or 'cpu'
        compute_type: 'float16' or 'int8'
        batch_size: Batch size for transcription

    Returns:
        dict with word-level lyrics and timestamps
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Transcribe with Whisper
    print(f"Loading WhisperX model...")
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    print(f"Transcribing: {audio_file}")
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)

    # 2. Align whisper output (word-level timestamps)
    print("Aligning words...")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # 3. Extract word-level data
    words = []
    for segment in result["segments"]:
        for word_data in segment.get("words", []):
            words.append({
                'word': word_data['word'].strip().lower(),
                'start': word_data.get('start', 0),
                'end': word_data.get('end', 0),
                'score': word_data.get('score', 0)
            })

    # 4. Save lyrics JSON
    lyrics = {
        'audio_file': audio_file,
        'language': result.get("language", "en"),
        'words': words,
        'segments': result["segments"]
    }
    lyrics_path = os.path.join(output_dir, 'lyrics.json')
    with open(lyrics_path, 'w') as f:
        json.dump(lyrics, f, indent=2)

    print(f"Transcription complete: {len(words)} words")
    print(f"Lyrics saved to {lyrics_path}")
    return lyrics


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python create_song_subtitles.py <vocals_audio_file> [output_dir]")
        sys.exit(1)

    audio_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else './output'
    transcribe_lyrics(audio_file, output_dir)
