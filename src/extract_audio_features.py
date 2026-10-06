import os
import glob
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from scipy.fft import rfft, rfftfreq
from tqdm import tqdm

def compute_mfcc(y, sr=16000, n_mfcc=13, n_fft=512, hop_length=256, n_mels=26):
    """Compute MFCCs using numpy/scipy."""
    # Pre-emphasis
    y_pre = np.append(y[0], y[1:] - 0.97 * y[:-1])
    
    # Framing
    num_frames = 1 + int((len(y_pre) - n_fft) / hop_length)
    if num_frames <= 0:
        return np.zeros((n_mfcc, 1))
    
    pad_len = (num_frames - 1) * hop_length + n_fft
    y_pad = np.pad(y_pre, (0, max(0, pad_len - len(y_pre))), mode='constant')
    
    frames = np.lib.stride_tricks.as_strided(
        y_pad,
        shape=(num_frames, n_fft),
        strides=(y_pad.strides[0] * hop_length, y_pad.strides[0])
    )
    
    # Windowing
    frames = frames * np.hamming(n_fft)
    
    # Power spectrum
    mag = np.abs(rfft(frames, n=n_fft, axis=-1))
    pow_frames = (1.0 / n_fft) * (mag ** 2)
    
    # Mel filterbank
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sr / 2) / 700)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    
    fbank = np.zeros((n_mels, int(n_fft / 2 + 1)))
    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin_points[m - 1]) / max(1, (f_m - f_m_minus))
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin_points[m + 1] - k) / max(1, (f_m_plus - f_m))
            
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    
    # DCT (Type II)
    num_ceps = n_mfcc
    n = np.arange(n_mels)
    k = np.arange(num_ceps)[:, np.newaxis]
    dct_basis = np.cos(np.pi * k * (2 * n + 1) / (2 * n_mels))
    mfcc = np.dot(filter_banks, dct_basis.T)
    return mfcc.T  # (n_mfcc, num_frames)

def compute_deltas(mfcc):
    """Compute delta features."""
    if mfcc.shape[1] < 3:
        return np.zeros_like(mfcc)
    delta = np.zeros_like(mfcc)
    delta[:, 1:-1] = (mfcc[:, 2:] - mfcc[:, :-2]) / 2.0
    delta[:, 0] = mfcc[:, 1] - mfcc[:, 0]
    delta[:, -1] = mfcc[:, -1] - mfcc[:, -2]
    return delta

def estimate_pitch_autocorr(y, sr=16000, frame_len=512, hop_len=256, fmin=60, fmax=400):
    """Estimate fundamental frequency (F0) using autocorrelation."""
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    pitches = []
    
    num_frames = 1 + int((len(y) - frame_len) / hop_len)
    if num_frames <= 0:
        return np.array([0.0]), 0.0
    
    for i in range(0, min(num_frames, 500)):  # sample up to 500 frames for speed
        start = i * hop_len
        frame = y[start:start + frame_len]
        if np.std(frame) < 1e-4:
            continue
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(frame) - 1:]
        if len(corr) > max_lag:
            peak_lag = min_lag + np.argmax(corr[min_lag:max_lag])
            r_max = corr[peak_lag]
            if r_max > 0.3 * corr[0]:
                pitches.append(sr / peak_lag)
                
    if len(pitches) == 0:
        return np.array([0.0]), 0.0
    pitches = np.array(pitches)
    voiced_ratio = len(pitches) / max(1, num_frames)
    return pitches, voiced_ratio

def extract_features_from_audio(file_path):
    """Extract full set of acoustic and prosodic features from a WAV file."""
    y, sr = sf.read(file_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    duration = len(y) / float(sr)
    
    # Basic energy & amplitude
    rms_frame_len = 512
    hop = 256
    num_frames = max(1, int(len(y) / hop))
    rms = np.sqrt(np.convolve(y**2, np.ones(rms_frame_len)/rms_frame_len, mode='valid'))
    if len(rms) == 0:
        rms = np.array([0.0])
        
    zcr = np.mean(np.abs(np.diff(np.sign(y)))) / 2.0
    abs_y = np.abs(y)
    mean_amp = float(np.mean(abs_y))
    std_amp = float(np.std(y))
    peak_amp = float(np.max(abs_y)) if len(abs_y) > 0 else 0.0
    silence_ratio = float(np.mean(rms < 0.01))
    
    # Pitch / F0
    pitches, voiced_ratio = estimate_pitch_autocorr(y, sr)
    f0_mean = float(np.mean(pitches))
    f0_std = float(np.std(pitches))
    f0_min = float(np.min(pitches))
    f0_max = float(np.max(pitches))
    f0_range = f0_max - f0_min
    
    # MFCCs & deltas
    mfcc = compute_mfcc(y, sr=sr, n_mfcc=13)
    d_mfcc = compute_deltas(mfcc)
    dd_mfcc = compute_deltas(d_mfcc)
    
    feat = {
        'audio_duration': duration,
        'rms_mean': float(np.mean(rms)),
        'rms_std': float(np.std(rms)),
        'rms_max': float(np.max(rms)),
        'zcr': zcr,
        'mean_amplitude': mean_amp,
        'std_amplitude': std_amp,
        'peak_amplitude': peak_amp,
        'silence_ratio': silence_ratio,
        'f0_mean': f0_mean,
        'f0_std': f0_std,
        'f0_min': f0_min,
        'f0_max': f0_max,
        'f0_range': f0_range,
        'voiced_ratio': voiced_ratio,
    }
    
    # Aggregate MFCC statistics
    for i in range(13):
        feat[f'mfcc_{i}_mean'] = float(np.mean(mfcc[i]))
        feat[f'mfcc_{i}_std'] = float(np.std(mfcc[i]))
        feat[f'mfcc_{i}_max'] = float(np.max(mfcc[i]))
        feat[f'mfcc_{i}_min'] = float(np.min(mfcc[i]))
        feat[f'd_mfcc_{i}_mean'] = float(np.mean(d_mfcc[i]))
        feat[f'd_mfcc_{i}_std'] = float(np.std(d_mfcc[i]))
        feat[f'dd_mfcc_{i}_mean'] = float(np.mean(dd_mfcc[i]))
        feat[f'dd_mfcc_{i}_std'] = float(np.std(dd_mfcc[i]))
        
    return feat

def process_dataset(csv_path, audio_dir, out_path):
    print(f"Extracting audio features for {csv_path}...")
    df = pd.read_csv(csv_path)
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        fname = row['filename']
        fpath = os.path.join(audio_dir, fname)
        if not os.path.exists(fpath):
            print(f"Warning: File not found {fpath}")
            continue
        feat = extract_features_from_audio(fpath)
        feat['filename'] = fname
        if 'label' in row:
            feat['label'] = row['label']
        records.append(feat)
        
    feat_df = pd.DataFrame(records)
    feat_df.to_csv(out_path, index=False)
    print(f"Saved {len(feat_df)} rows of audio features to {out_path} (shape: {feat_df.shape})")
    return feat_df

if __name__ == '__main__':
    train_feat = process_dataset(
        'Dataset_Final/train.csv',
        'Dataset_Final/train',
        'artifacts/audio_features/train_audio_features.csv'
    )
    test_feat = process_dataset(
        'Dataset_Final/test.csv',
        'Dataset_Final/test',
        'artifacts/audio_features/test_audio_features.csv'
    )
    print("All audio features extracted successfully!")
