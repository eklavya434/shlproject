import os
import glob
import pandas as pd
import soundfile as sf
import numpy as np
import whisper
from tqdm import tqdm

def transcribe_dataset(csv_path, audio_dir, out_path, model):
    df = pd.read_csv(csv_path)
    
    # Check if existing transcripts file exists for resumption
    if os.path.exists(out_path):
        existing_df = pd.read_csv(out_path)
        done_files = set(existing_df['filename'].dropna())
        print(f"Resuming {out_path}: {len(done_files)} files already transcribed.")
        records = existing_df.to_dict('records')
    else:
        done_files = set()
        records = []
        
    remaining = df[~df['filename'].isin(done_files)]
    print(f"Transcribing {len(remaining)} remaining files for {csv_path}...")
    
    count = 0
    for _, row in tqdm(remaining.iterrows(), total=len(remaining)):
        fname = row['filename']
        fpath = os.path.join(audio_dir, fname)
        if not os.path.exists(fpath):
            print(f"Warning: {fpath} not found.")
            continue
            
        try:
            audio, sr = sf.read(fpath)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            duration = len(audio) / float(sr)
            audio = audio.astype(np.float32)
            
            result = model.transcribe(audio, fp16=False, language='en')
            transcript = result['text'].strip()
            
            rec = {
                'filename': fname,
                'transcript': transcript,
                'duration': round(duration, 2),
                'word_count': len(transcript.split())
            }
            if 'label' in row:
                rec['label'] = row['label']
            records.append(rec)
            count += 1
            
            # Save checkpoint every 25 files
            if count % 25 == 0:
                pd.DataFrame(records).to_csv(out_path, index=False)
        except Exception as e:
            print(f"Error transcribing {fname}: {e}")
            rec = {
                'filename': fname,
                'transcript': '',
                'duration': 0.0,
                'word_count': 0
            }
            if 'label' in row:
                rec['label'] = row['label']
            records.append(rec)
            
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"Finished {csv_path}! Total transcribed: {len(records)} saved to {out_path}")

if __name__ == '__main__':
    print("Loading Whisper tiny.en model...")
    model = whisper.load_model('tiny.en')
    
    # Process test set first (smaller: 216 files ~ 5-6 minutes)
    transcribe_dataset(
        'Dataset_Final/test.csv',
        'Dataset_Final/test',
        'artifacts/transcripts/test_transcripts.csv',
        model
    )
    
    # Process train set (769 files ~ 18-20 minutes)
    transcribe_dataset(
        'Dataset_Final/train.csv',
        'Dataset_Final/train',
        'artifacts/transcripts/train_transcripts.csv',
        model
    )
    print("All transcriptions complete!")
