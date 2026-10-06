import os
import re
import math
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from tqdm import tqdm

FILLER_WORDS = set(['um', 'uh', 'er', 'ah', 'like', 'well', 'so', 'basically', 'actually', 'mean', 'you know'])

def count_syllables(word):
    """Simple heuristic syllable counter."""
    word = word.lower()
    if len(word) <= 3:
        return 1
    word = re.sub(r'(?:[^laeiouy]|ed|es|e)$', '', word)
    word = re.sub(r'^y', '', word)
    matches = re.findall(r'[aeiouy]{1,2}', word)
    return max(1, len(matches))

def extract_linguistic_features_from_text(text, duration=60.0):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {
            'num_words': 0, 'num_chars': 0, 'num_sentences': 0,
            'avg_words_per_sentence': 0.0, 'median_sentence_length': 0.0,
            'max_sentence_length': 0.0, 'min_sentence_length': 0.0, 'std_sentence_length': 0.0,
            'unique_words': 0, 'ttr': 0.0, 'root_ttr': 0.0, 'corrected_ttr': 0.0, 'bilog_ttr': 0.0,
            'avg_word_length': 0.0, 'lexical_density': 0.0,
            'pos_noun_ratio': 0.0, 'pos_verb_ratio': 0.0, 'pos_adj_ratio': 0.0, 'pos_adv_ratio': 0.0,
            'pos_pron_ratio': 0.0, 'pos_det_ratio': 0.0, 'pos_prep_ratio': 0.0, 'pos_conj_ratio': 0.0, 'pos_modal_ratio': 0.0,
            'speech_rate_wps': 0.0, 'syllable_rate_sps': 0.0,
            'repeated_word_ratio': 0.0, 'filler_word_ratio': 0.0,
            'short_sentence_ratio': 0.0, 'long_sentence_ratio': 0.0, 'fragment_ratio': 0.0
        }

    raw_words = word_tokenize(text)
    words = [w.lower() for w in raw_words if w.isalnum()]
    N = len(words)
    chars = sum(len(w) for w in words)
    
    sentences = sent_tokenize(text)
    num_sents = max(1, len(sentences))
    sent_lens = [len([w for w in word_tokenize(s) if w.isalnum()]) for s in sentences]
    sent_lens = [sl for sl in sent_lens if sl > 0]
    if len(sent_lens) == 0:
        sent_lens = [max(1, N)]
        
    avg_words_per_sent = float(np.mean(sent_lens))
    med_words_per_sent = float(np.median(sent_lens))
    max_words_per_sent = float(np.max(sent_lens))
    min_words_per_sent = float(np.min(sent_lens))
    std_words_per_sent = float(np.std(sent_lens))
    
    # Vocabulary diversity
    vocab = set(words)
    V = len(vocab)
    ttr = V / max(1, N)
    root_ttr = V / math.sqrt(max(1, N))
    corrected_ttr = V / math.sqrt(max(1, 2 * N))
    bilog_ttr = (math.log(max(1, V)) / math.log(max(2, N))) if N > 1 else 1.0
    avg_word_len = chars / max(1, N)
    
    # Repeated adjacent words
    repeated_count = sum(1 for i in range(len(words) - 1) if words[i] == words[i+1])
    repeated_ratio = repeated_count / max(1, N)
    
    # Filler words
    filler_count = sum(1 for w in words if w in FILLER_WORDS)
    filler_ratio = filler_count / max(1, N)
    
    # POS tagging
    pos_tags = pos_tag(raw_words)
    noun_count = sum(1 for _, t in pos_tags if t.startswith('NN'))
    verb_count = sum(1 for _, t in pos_tags if t.startswith('VB'))
    adj_count = sum(1 for _, t in pos_tags if t.startswith('JJ'))
    adv_count = sum(1 for _, t in pos_tags if t.startswith('RB'))
    pron_count = sum(1 for _, t in pos_tags if t.startswith('PRP'))
    det_count = sum(1 for _, t in pos_tags if t.startswith('DT'))
    prep_count = sum(1 for _, t in pos_tags if t in ('IN', 'TO'))
    conj_count = sum(1 for _, t in pos_tags if t.startswith('CC'))
    modal_count = sum(1 for _, t in pos_tags if t == 'MD')
    
    total_tags = max(1, len(pos_tags))
    content_words = noun_count + verb_count + adj_count + adv_count
    lexical_density = content_words / max(1, N)
    
    # Sentence characteristics
    short_sents = sum(1 for sl in sent_lens if sl < 4)
    long_sents = sum(1 for sl in sent_lens if sl > 25)
    
    # Sentence fragments (sentences without any verb)
    frag_count = 0
    for s in sentences:
        s_tags = [t for _, t in pos_tag(word_tokenize(s))]
        if not any(t.startswith('VB') for t in s_tags):
            frag_count += 1
            
    # Speech rates
    dur = max(1.0, float(duration))
    wps = N / dur
    syllables = sum(count_syllables(w) for w in words)
    sps = syllables / dur
    
    return {
        'num_words': N,
        'num_chars': chars,
        'num_sentences': len(sentences),
        'avg_words_per_sentence': avg_words_per_sent,
        'median_sentence_length': med_words_per_sent,
        'max_sentence_length': max_words_per_sent,
        'min_sentence_length': min_words_per_sent,
        'std_sentence_length': std_words_per_sent,
        'unique_words': V,
        'ttr': ttr,
        'root_ttr': root_ttr,
        'corrected_ttr': corrected_ttr,
        'bilog_ttr': bilog_ttr,
        'avg_word_length': avg_word_len,
        'lexical_density': lexical_density,
        'pos_noun_ratio': noun_count / total_tags,
        'pos_verb_ratio': verb_count / total_tags,
        'pos_adj_ratio': adj_count / total_tags,
        'pos_adv_ratio': adv_count / total_tags,
        'pos_pron_ratio': pron_count / total_tags,
        'pos_det_ratio': det_count / total_tags,
        'pos_prep_ratio': prep_count / total_tags,
        'pos_conj_ratio': conj_count / total_tags,
        'pos_modal_ratio': modal_count / total_tags,
        'speech_rate_wps': wps,
        'syllable_rate_sps': sps,
        'repeated_word_ratio': repeated_ratio,
        'filler_word_ratio': filler_ratio,
        'short_sentence_ratio': short_sents / num_sents,
        'long_sentence_ratio': long_sents / num_sents,
        'fragment_ratio': frag_count / num_sents
    }

def process_transcripts(transcripts_csv, out_csv):
    df = pd.read_csv(transcripts_csv)
    print(f"Extracting linguistic features for {transcripts_csv} ({len(df)} rows)...")
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row['transcript']) if pd.notna(row['transcript']) else ''
        dur = row['duration'] if 'duration' in row and pd.notna(row['duration']) else 60.0
        feat = extract_linguistic_features_from_text(text, dur)
        feat['filename'] = row['filename']
        if 'label' in row:
            feat['label'] = row['label']
        records.append(feat)
        
    feat_df = pd.DataFrame(records)
    feat_df.to_csv(out_csv, index=False)
    print(f"Saved {len(feat_df)} linguistic feature rows to {out_csv} (shape: {feat_df.shape})")
    return feat_df

if __name__ == '__main__':
    os.makedirs('artifacts/linguistic_features', exist_ok=True)
    if os.path.exists('artifacts/transcripts/test_transcripts.csv'):
        process_transcripts(
            'artifacts/transcripts/test_transcripts.csv',
            'artifacts/linguistic_features/test_linguistic_features.csv'
        )
    if os.path.exists('artifacts/transcripts/train_transcripts.csv'):
        process_transcripts(
            'artifacts/transcripts/train_transcripts.csv',
            'artifacts/linguistic_features/train_linguistic_features.csv'
        )
