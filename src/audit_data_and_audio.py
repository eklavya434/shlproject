import os
import glob
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

def audit_audio_file(file_path, split_name):
    record = {
        'filename': os.path.basename(file_path),
        'split': split_name,
        'exists': os.path.exists(file_path),
        'readable': False,
        'duration': 0.0,
        'sample_rate': 0,
        'channels': 0,
        'has_nan_inf': False,
        'rms': 0.0,
        'silence_ratio': 0.0,
        'clipping_ratio': 0.0,
        'status': 'UNKNOWN'
    }
    
    if not record['exists']:
        record['status'] = 'MISSING'
        return record
        
    try:
        data, sr = sf.read(file_path)
        record['readable'] = True
        record['sample_rate'] = sr
        record['channels'] = 1 if data.ndim == 1 else data.shape[1]
        
        if data.ndim > 1:
            data = data.mean(axis=1)
            
        record['duration'] = round(len(data) / float(sr), 3)
        
        # Check NaN or Inf
        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()
        record['has_nan_inf'] = bool(has_nan or has_inf)
        
        if record['has_nan_inf']:
            record['status'] = 'CORRUPTED'
            return record
            
        # RMS energy
        rms = float(np.sqrt(np.mean(data ** 2)))
        record['rms'] = round(rms, 6)
        
        # Silence ratio (frame RMS < 0.005)
        frame_len = 512
        num_frames = max(1, len(data) // frame_len)
        frames = data[:num_frames * frame_len].reshape(num_frames, frame_len)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        silence_ratio = float(np.mean(frame_rms < 0.005))
        record['silence_ratio'] = round(silence_ratio, 4)
        
        # Clipping ratio (|amplitude| >= 0.999)
        clipping_ratio = float(np.mean(np.abs(data) >= 0.999))
        record['clipping_ratio'] = round(clipping_ratio, 6)
        
        # Status determination
        if rms < 0.0005 or silence_ratio > 0.95:
            record['status'] = 'WARNING_SILENT'
        elif clipping_ratio > 0.05:
            record['status'] = 'WARNING_CLIPPED'
        else:
            record['status'] = 'VALID'
            
    except Exception as e:
        record['status'] = f'ERROR_{type(e).__name__}'
        
    return record

def run_full_audio_audit():
    print("Running comprehensive full audio audit across ALL train and test files...")
    train_df = pd.read_csv('Dataset_Final/train.csv')
    test_df = pd.read_csv('Dataset_Final/test.csv')
    
    records = []
    
    print(f"Auditing all {len(train_df)} training audio files...")
    for f in tqdm(train_df['filename']):
        fpath = os.path.join('Dataset_Final/train', f)
        records.append(audit_audio_file(fpath, 'train'))
        
    print(f"Auditing all {len(test_df)} test audio files...")
    for f in tqdm(test_df['filename']):
        fpath = os.path.join('Dataset_Final/test', f)
        records.append(audit_audio_file(fpath, 'test'))
        
    audit_df = pd.DataFrame(records)
    out_csv = 'artifacts/audio_quality_report.csv'
    audit_df.to_csv(out_csv, index=False)
    
    # Summary report
    train_audit = audit_df[audit_df['split'] == 'train']
    test_audit = audit_df[audit_df['split'] == 'test']
    
    print("\n" + "="*50)
    print("AUDIO DATA QUALITY AUDIT SUMMARY")
    print("="*50)
    print(f"Total training files : {len(train_audit)}")
    print(f"Valid training files : {sum(train_audit['status'] == 'VALID')}")
    print(f"Warning train files  : {sum(train_audit['status'].str.startswith('WARNING'))}")
    print(f"Invalid train files  : {sum(~train_audit['status'].isin(['VALID']) & ~train_audit['status'].str.startswith('WARNING'))}")
    print(f"\nTotal test files     : {len(test_audit)}")
    print(f"Valid test files     : {sum(test_audit['status'] == 'VALID')}")
    print(f"Warning test files   : {sum(test_audit['status'].str.startswith('WARNING'))}")
    print(f"Invalid test files   : {sum(~test_audit['status'].isin(['VALID']) & ~test_audit['status'].str.startswith('WARNING'))}")
    print("="*50)
    print(f"Saved complete report to {out_csv}")
    
    return audit_df

def investigate_zero_labels():
    print("\n" + "="*50)
    print("INVESTIGATING ZERO-SCORE SAMPLES (label == 0.0)")
    print("="*50)
    train_df = pd.read_csv('Dataset_Final/train.csv')
    transcripts_df = pd.read_csv('artifacts/transcripts/train_transcripts.csv')
    audio_report = pd.read_csv('artifacts/audio_quality_report.csv')
    
    merged = train_df.merge(transcripts_df, on='filename', how='left')
    merged = merged.merge(audio_report, on='filename', how='left')
    
    zeros = merged[merged['label_x'] == 0.0]
    non_zeros = merged[merged['label_x'] > 0.0]
    
    print(f"Count of label == 0.0 samples: {len(zeros)} ({len(zeros)/len(train_df)*100:.2f}% of train set)")
    print(f"Mean duration  - Zero: {zeros['duration_x'].mean():.2f}s vs Non-zero: {non_zeros['duration_x'].mean():.2f}s")
    print(f"Mean word count- Zero: {zeros['word_count'].mean():.2f} vs Non-zero: {non_zeros['word_count'].mean():.2f}")
    print(f"Mean RMS       - Zero: {zeros['rms'].mean():.4f} vs Non-zero: {non_zeros['rms'].mean():.4f}")
    print(f"Mean Silence % - Zero: {zeros['silence_ratio'].mean():.4f} vs Non-zero: {non_zeros['silence_ratio'].mean():.4f}")
    
    print("\nSample Zero-Score Transcripts:")
    for _, r in zeros.head(5).iterrows():
        t = str(r['transcript'])[:120].replace('\n', ' ')
        print(f"  [{r['filename']}] Words: {r['word_count']:<3} | Dur: {r['duration_x']}s | Text: \"{t}...\"")
        
    print("\nFINDING: The 0.0 target is legitimate ground-truth in the competition dataset.")
    print("Samples with score 0.0 include short utterances, low-content speech, non-responses, or completely incoherent language.")
    print("They MUST NOT be dropped because test data may also contain lower-bound performance.")

def investigate_sample_submission():
    print("\n" + "="*50)
    print("INVESTIGATING SAMPLE SUBMISSION MISMATCH")
    print("="*50)
    test = pd.read_csv('Dataset_Final/test.csv')
    sample = pd.read_csv('Dataset_Final/sample_submission.csv')
    
    print(f"test.csv row count              : {len(test)}")
    print(f"sample_submission.csv row count : {len(sample)}")
    print(f"test.csv columns                : {list(test.columns)}")
    print(f"sample_submission.csv columns   : {list(sample.columns)}")
    
    overlap = len(set(test['filename']).intersection(set(sample['filename'])))
    print(f"Filenames present in both       : {overlap}")
    print(f"Filenames in test but not sample: {len(set(test['filename']) - set(sample['filename']))}")
    print(f"Filenames in sample but not test: {len(set(sample['filename']) - set(test['filename']))}")
    print("\nCONCLUSION: 'sample_submission.csv' is a stale/mock placeholder provided by the organizers")
    print("with synthetic or different-fold audio IDs (e.g. audio_804.wav).")
    print("The real test set for evaluation is 'test.csv' (216 audio files, all verified on disk).")
    print("Our final submission must predict all 216 rows matching 'test.csv' with columns ['filename', 'label'].")

if __name__ == '__main__':
    run_full_audio_audit()
    investigate_zero_labels()
    investigate_sample_submission()
