import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def extract_embeddings(transcripts_csv, out_npy_path, model):
    df = pd.read_csv(transcripts_csv)
    print(f"Extracting sentence embeddings for {transcripts_csv} ({len(df)} rows)...")
    texts = df['transcript'].fillna('').astype(str).tolist()
    # Batch encode with normalization
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)
    np.save(out_npy_path, embeddings)
    print(f"Saved embeddings to {out_npy_path} with shape {embeddings.shape}")
    return embeddings

if __name__ == '__main__':
    os.makedirs('artifacts/text_embeddings', exist_ok=True)
    print("Loading SentenceTransformer all-MiniLM-L6-v2...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    if os.path.exists('artifacts/transcripts/test_transcripts.csv'):
        extract_embeddings(
            'artifacts/transcripts/test_transcripts.csv',
            'artifacts/text_embeddings/test_text_embeddings.npy',
            model
        )
    if os.path.exists('artifacts/transcripts/train_transcripts.csv'):
        extract_embeddings(
            'artifacts/transcripts/train_transcripts.csv',
            'artifacts/text_embeddings/train_text_embeddings.npy',
            model
        )
