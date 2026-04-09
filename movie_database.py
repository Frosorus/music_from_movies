import whisperx
import pysrt
import html
import re
import string
import glob
import json


def clean_subtitle_text(text):
    # 1. Décoder les entités HTML (&nbsp;, &amp;, etc.)
    text = html.unescape(text)

    # 2. Supprimer toutes les balises HTML/XML : <i>, </i>, <font color="...">, etc.
    text = re.sub(r"</?[^>]+>", "", text)

    # 3. Supprimer les balises de style SSA/ASS comme {\an8}, {\b1}, etc.
    text = re.sub(r"{\\.*?}", "", text)

    # 4. Remplacer les sauts de ligne par des espaces
    text = text.replace("\n", " ")

    # 5. Supprimer toute la ponctuation
    text = text.translate(str.maketrans('', '', string.punctuation + "«»“”‘’"))

    # 6. Supprimer les espaces en double
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()

device = "cuda"
language = "en"
compute_type = "float16"
batch_size=16
audio_files = "/mnt/e/extracted_english/*.wav"


for audio_file in glob.glob(audio_files):
    print(audio_file)  
    try:
        with open('movie_aligned_en.json', 'r') as fp:
            dictionnary_word = json.load(fp)
    except :
        dictionnary_word = {}
    if audio_file not in dictionnary_word.keys():
        subtitle_file = audio_file.replace("audio.wav", "subs.srt")
        subs = pysrt.open(subtitle_file)

        segments = [{"text": clean_subtitle_text(sub.text.strip()), 
                    "start" : sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds + sub.start.milliseconds / 1000 -30 if sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds + sub.start.milliseconds / 1000 -30 >=0 else 0 , 
                    "end":sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds + sub.end.milliseconds / 1000 +30 } for sub in subs]

        # # 1. Transcribe with original whisper (batched)
        # # export LD_LIBRARY=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/srt-lang-detect/lib/python3.11/site-packages/nvidia/cublas/lib:$HOME/miniconda3/envs/srt-lang-detect/lib/python3.11/site-packages/nvidia/cudnn/lib
        # model = whisperx.load_model("large-v2", device, compute_type=compute_type)

        # # save model to local path (optional)
        # # model_dir = "/path/"
        # # model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

        # audio = whisperx.load_audio(audio_file)
        # result = model.transcribe(audio, batch_size=batch_size)

        # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code="en", device=device)

        result = whisperx.align(segments, model_a, metadata, audio_file, device, return_char_alignments=False)

        word_list = [loop['words'] for loop in result['segments']]
        word_list_concat = sum(word_list, [])
        print(audio_file)
        dictionnary_word[audio_file] = word_list_concat
        with open('movie_aligned_en.json', 'w') as fp:
            json.dump(dictionnary_word, fp)