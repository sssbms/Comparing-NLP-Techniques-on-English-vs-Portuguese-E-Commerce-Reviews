# task2_language_model.py


import pandas as pd
from nltk import bigrams
from collections import Counter
import random
import ast

# =============== #
# 1. Load datasets
# =============== #

english_df = pd.read_csv("/Users/apple/Desktop/KD04403-NLP/ASSIGN/english_cleaned.csv")
brazilian_df = pd.read_csv("/Users/apple/Desktop/KD04403-NLP/ASSIGN/brazilian_cleaned.csv")

# Convert 'cleaned_tokens' string to list
english_df['cleaned_tokens'] = english_df['cleaned_tokens'].apply(ast.literal_eval)
brazilian_df['cleaned_tokens'] = brazilian_df['cleaned_tokens'].apply(ast.literal_eval)

# Combine all tokens into one list for each language
english_tokens = [word for tokens in english_df['cleaned_tokens'] for word in tokens]
brazilian_tokens = [word for tokens in brazilian_df['cleaned_tokens'] for word in tokens]

# ============================================ #
# 2. Build Bigram Counts and Vocabulary (English)
# ============================================ #
def build_bigram_model(tokens):
    vocab = set(tokens)
    V = len(vocab)
    bigram_list = list(bigrams(tokens))
    bigram_counts = Counter(bigram_list)
    unigram_counts = Counter(tokens)
    return bigram_counts, unigram_counts, V

eng_bigram_counts, eng_unigram_counts, eng_V = build_bigram_model(english_tokens)
pt_bigram_counts, pt_unigram_counts, pt_V = build_bigram_model(brazilian_tokens)

# ==================================================== #
# 3. Add-One (Laplace) Smoothing + Save Probabilities
# ==================================================== #
def bigram_probability(w1, w2, bigram_counts, unigram_counts, V):
    """Calculate Laplace-smoothed probability for a bigram."""
    return (bigram_counts[(w1, w2)] + 1) / (unigram_counts[w1] + V)

def calculate_and_save_probabilities(tokens, bigram_counts, unigram_counts, V, filename):
    """Calculate raw and smoothed probabilities for all bigrams and save to CSV."""
    data = []
    for (w1, w2), count in bigram_counts.items():
        raw_prob = count / unigram_counts[w1]
        smooth_prob = bigram_probability(w1, w2, bigram_counts, unigram_counts, V)
        data.append({
            "Word_1": w1,
            "Word_2": w2,
            "Bigram_Count": count,
            "Word_1_Count": unigram_counts[w1],
            "Raw_Probability": round(raw_prob, 6),
            "Smoothed_Probability": round(smooth_prob, 6)
        })

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f" Saved bigram probabilities to {filename}")
    return df

# Calculate and save for both languages
eng_probs = calculate_and_save_probabilities(
    english_tokens, eng_bigram_counts, eng_unigram_counts, eng_V, "english_bigram_probabilities.csv"
)
pt_probs = calculate_and_save_probabilities(
    brazilian_tokens, pt_bigram_counts, pt_unigram_counts, pt_V, "brazilian_bigram_probabilities.csv"
)

# ================================================= #
# 4. Display Sample Probabilities (for Report Table)
# ================================================= #
def sample_probabilities(tokens, bigram_counts, unigram_counts, V, n=5):
    pairs = list(bigrams(tokens))[:n]
    for w1, w2 in pairs:
        raw = bigram_counts[(w1, w2)] / unigram_counts[w1]
        smooth = bigram_probability(w1, w2, bigram_counts, unigram_counts, V)
        print(f"Bigram: ({w1}, {w2}) | Raw: {raw:.4f} | Smoothed: {smooth:.4f}")

print("\n English Sample Bigram Probabilities:")
sample_probabilities(english_tokens, eng_bigram_counts, eng_unigram_counts, eng_V)

print("\n Portuguese Sample Bigram Probabilities:")
sample_probabilities(brazilian_tokens, pt_bigram_counts, pt_unigram_counts, pt_V)

# ============================================= #
# 5. Generate Sentences (Based on Highest Prob.)
# ============================================= #
def generate_sentence(bigram_counts, unigram_counts, V, max_len=8):
    sentence = []
    current_word = random.choice(list(unigram_counts.keys()))
    sentence.append(current_word)

    for _ in range(max_len - 1):
        # find most probable next word
        candidates = [(w2, bigram_probability(current_word, w2, bigram_counts, unigram_counts, V))
                      for (w1, w2) in bigram_counts.keys() if w1 == current_word]
        if not candidates:
            break
        next_word = max(candidates, key=lambda x: x[1])[0]
        sentence.append(next_word)
        current_word = next_word

    return ' '.join(sentence)

print("\n Generated Sentences (English):")
for i in range(5):
    print(f"{i+1}.", generate_sentence(eng_bigram_counts, eng_unigram_counts, eng_V))

print("\n Generated Sentences (Portuguese):")
for i in range(5):
    print(f"{i+1}.", generate_sentence(pt_bigram_counts, pt_unigram_counts, pt_V))

print("\nTask 2 completed successfully! Bigram model + Laplace smoothing applied.")
