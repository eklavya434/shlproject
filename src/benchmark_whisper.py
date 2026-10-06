import time
import glob
import numpy as np
import soundfile as sf
import whisper

train_files = glob.glob('Dataset_Final/train/*.wav')[:5]
print("Loading Whisper model tiny.en...")
model = whisper.load_model('tiny.en')

t0 = time.time()
for i, f in enumerate(train_files):
    t_start = time.time()
    audio, sr = sf.read(f)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    res = model.transcribe(audio, fp16=False, language='en')
    dur = time.time() - t_start
    print(f"Sample {i+1} ({f}): audio_len={len(audio)/sr:.1f}s, transcribed in {dur:.2f}s, words={len(res['text'].split())}")

total_t = time.time() - t0
print(f"Total time for 5 files: {total_t:.2f}s, avg: {total_t/5:.2f}s per file")
