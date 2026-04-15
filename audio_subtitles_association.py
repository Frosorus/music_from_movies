import subprocess
import json
from pathlib import Path

TEXT_SUBTITLE_CODECS = {"subrip", "mov_text", "ass", "ssa", "webvtt"}
SKIP_TITLE_KEYWORDS = {"forced", "commentary"}
ENGLISH_TITLE_KEYWORDS = {"english", "anglais"}

def run_ffprobe(file_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=index,codec_type,codec_name:stream_tags=language,title",
        "-of", "json",
        str(file_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout)["streams"]

def find_english_audio(streams):
    candidates = [
        s for s in streams
        if s.get("codec_type") == "audio" and s.get("tags", {}).get("language") == "eng"
    ]
    if not candidates:
        return None
    # Prefer the track with the most channels (best quality)
    return max(candidates, key=lambda s: s.get("channels", 0))["index"]

def find_english_subtitle(streams):
    candidates = []
    for s in streams:
        if s.get("codec_type") != "subtitle":
            continue
        if s.get("codec_name", "") not in TEXT_SUBTITLE_CODECS:
            continue
        lang = s.get("tags", {}).get("language", "")
        title = s.get("tags", {}).get("title", "").lower()
        is_english = lang == "eng" or any(kw in title for kw in ENGLISH_TITLE_KEYWORDS)
        if not is_english:
            continue
        is_sdh = "sdh" in title or "hearing impaired" in title
        is_skipped = any(kw in title for kw in SKIP_TITLE_KEYWORDS)
        if is_skipped:
            continue
        candidates.append((s["index"], title, is_sdh))

    # Prefer non-SDH; fall back to SDH if nothing else available
    preferred = [(i, t) for i, t, sdh in candidates if not sdh]
    fallback   = [(i, t) for i, t, sdh in candidates if sdh]
    pool = preferred if preferred else fallback

    if not pool:
        return None, None

    # Prefer tracks labelled "full", "complet", or "anglais"
    for index, title in pool:
        if any(kw in title for kw in ("full", "complet", "anglais")):
            return index, title

    return pool[0]

def process_video(file_path):
    try:
        streams = run_ffprobe(file_path)
    except Exception as e:
        print(f"  Error probing {file_path}: {e}")
        return None
    audio_index = find_english_audio(streams)
    subtitle_index, subtitle_title = find_english_subtitle(streams)

    if audio_index is None:
        print(f"  Skipped (no English audio): {Path(file_path).name}")
    elif subtitle_index is None:
        print(f"  Skipped (no English subtitle): {Path(file_path).name}")

    return {
        "audio_stream_index": audio_index,
        "subtitle_stream_index": subtitle_index,
        "subtitle_title": subtitle_title
    }

def main(directory="."):
    output_path = Path("audio_subtitles.json")
    # Load existing results to allow incremental runs
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            output = json.load(f)
        print(f"Loaded {len(output)} existing entries.")
    else:
        output = {}

    video_exts = (".mkv", ".mp4", ".avi", ".mov")

    for path in sorted(Path(directory).rglob("*")):
        if path.suffix.lower() not in video_exts or not path.is_file():
            continue
        key = str(path)
        if key in output:
            continue  # already processed
        print(f"Scanning: {path.name}")
        info = process_video(path)
        if info:
            output[key] = info
            # Save after each movie so progress is never lost
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Done. {len(output)} movies in audio_subtitles.json.")

if __name__ == "__main__":
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    main(directory)
