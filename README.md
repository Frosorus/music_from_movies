## Music from Movies

Takes a YouTube song and assembles a video from matching movie clips, one clip per lyric word.

## How it works

### Pipeline

```
YouTube URL
    │
    ▼
1. Download song + auto-generated subtitles  (yt-dlp)
    │
    ▼
2. Separate vocals from instrumental         (Demucs)
    │
    ▼
3. Align lyrics word-by-word to timestamps   (stable-ts)
    │
    ▼
4. Match each word to a movie clip           (librosa + phonetic search)
    │
    ▼
5. Assemble final video                      (ffmpeg / MoviePy)
```

### Movie database

Before running the pipeline, a database must be built from a local film library.
For each movie, the English subtitles embedded in the MKV file are extracted, then
stable-ts aligns each word to an exact timestamp. Each entry is stored with audio
features (pitch, energy, zero-crossing rate) used later for scoring.

The pre-built database (~300 MB) is not included in this repo — open an issue or
contact me directly on GitHub to request it. See [Build the database](#build-the-database)
to create your own from a local film library.

## Setup

Two separate conda environments are required — their dependencies conflict and
cannot be installed together.

```bash
# Main environment (steps 0, 1, 3, 4)
conda create -n extract_song python=3.11
conda activate extract_song
pip install -r requirements_main.txt

# Alignment environment (step 2 + database population)
conda create -n stable-ts python=3.11
conda activate stable-ts
pip install -r requirements_stable_ts.txt
```

Then point `pipeline.sh` to each environment:

```bash
export EXTRACT_SONG_BIN="/path/to/miniconda3/envs/extract_song/bin/"
export STABLE_TS_PYTHON="/path/to/miniconda3/envs/stable-ts/bin/python"
```

## Usage

```bash
conda activate extract_song
./pipeline.sh <youtube_url> <output_dir>
```

The final video is written to `<output_dir>/<song_title>/final_video.mp4`.

## Build the database

```bash
# 1. Scan your film library for English audio and subtitle streams
conda activate extract_song
python audio_subtitles_association.py /path/to/films/

# 2. Populate the database (runs stable-ts on every movie — GPU recommended)
conda activate stable-ts
python populate_database.py movie_clips.db audio_subtitles.json
```

## Design choices

**Movie subtitles instead of transcription** — Rather than running speech-to-text
on each film, the pipeline uses the English subtitles already embedded in the MKV
files. This is significantly faster and produces cleaner word boundaries than
automatic transcription.

**YouTube auto-subtitles for lyrics** — WhisperX was tested for song transcription
but YouTube's auto-generated subtitles consistently produced better word-level
accuracy for music.

**stable-ts for word alignment** — WhisperX alignment was tried first. A CTC-aligner
was also tested but required patching the model code and produced random C++ crashes.
stable-ts was the most reliable option.

**Phonetic matching with Double Metaphone** — Exact string matching misses
homophones and spelling differences between lyrics and subtitles (e.g. *nite* vs
*night*). Phonetic indexing finds clips even when spellings diverge.

**Audio feature scoring** — Candidate clips are ranked by cosine similarity on a
vector of pitch, energy, and zero-crossing rate extracted from the source audio.
A per-movie usage counter prevents any single film from dominating the output.

## Known limitations

**Word alignment quality** — The main bottleneck of the pipeline. Timestamps
produced by stable-ts are often slightly off, causing the selected clip to be out
of sync with the song. This is an open problem.

**Sung vs. spoken words** — Matching a word held over a long sustained note to a
short spoken clip is a fundamental mismatch: the audio features are too different
in nature for the score function to bridge reliably.

**ffmpeg / Rubberband export bug** — A handful of clips per song fail to export
correctly due to an interaction between ffmpeg and the Rubberband time-stretcher.
Affected clips are dropped silently.

**Short clip crashes** — Audio feature extraction will fail on clips that are too
short. These are caught and skipped, but they result in missing words in the final
video.

**Score function** — The current feature vector and weighting are a first
approximation. The ranking does not consistently surface the best match and would
benefit from more careful tuning or a learned scoring model.

## Ideas / future work

- **Signal-level alignment** — Apply DTW or a similar technique on top of
  stable-ts timestamps to tighten word boundaries against the actual audio.
- **Audio morphing for sustained notes** — Investigate histogram matching or
  spectral distortion to better bridge sung and spoken word audio.
- **Better scoring** — Expand the feature set or train a small model to rank
  candidate clips.
- **App** — Turn the pipeline into a proper web or desktop application once
  output quality is good enough.
