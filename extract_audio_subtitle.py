import json
import subprocess
from pathlib import Path

def extract_audio_subtitle(video_file, audio_index, subtitle_index):
    file_name = Path(video_file).with_suffix("").name
    base = Path("/mnt/e/extracted_english/{}".format(file_name))

    if audio_index is not None:
        audio_out = base.with_suffix(".audio.wav")
        if not Path.exists(audio_out):
            cmd_audio = [
                "ffmpeg", "-y", "-i", video_file,
                "-map", f"0:{audio_index}", "-c:a", "pcm_s16le",
                str(audio_out)
            ]
            subprocess.run(cmd_audio, check=True)
            print(f"Audio extrait → {audio_out.name}")

    if subtitle_index is not None:
        subtitle_out = base.with_suffix(".subs.srt")
        if not Path.exists(subtitle_out):
            cmd_sub = [
                "ffmpeg", "-y", "-i", video_file,
                "-map", f"0:{subtitle_index}",
                str(subtitle_out)
            ]
            subprocess.run(cmd_sub, check=True)
            print(f"Sous-titres extraits → {subtitle_out.name}")

def main(json_path="audio_subtitles.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for video_file, info in data.items():
        audio_index = info.get("audio_stream_index")
        subtitle_index = info.get("subtitle_stream_index")
        print(f"Traitement de {video_file}...")
        if audio_index != None and subtitle_index != None :
            extract_audio_subtitle(video_file, audio_index, subtitle_index)

if __name__ == "__main__":
    main()
