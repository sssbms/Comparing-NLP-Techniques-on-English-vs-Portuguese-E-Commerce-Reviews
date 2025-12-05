# ============================================
# DATA CLEANING & TEXT PREPROCESSING
# Datasets: English + Brazilian Portuguese
# ============================================

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import stanza

# -------------------------------------------------------------------
# SETUP
# -------------------------------------------------------------------
nltk.download('stopwords')
nltk.download('punkt')

# Initialize Stanza for Portuguese
stanza.download('pt')
nlp_pt = stanza.Pipeline('pt', processors='tokenize')

# -------------------------------------------------------------------
# SECTION A: ENGLISH DATASET
# -------------------------------------------------------------------
print("\n🇬🇧 PREPROCESSING ENGLISH DATASET...")

# Load the English sample
english_path = '/Users/apple/Desktop/KD04403-NLP/ASSIGN/english_sample.csv'
df_en = pd.read_csv(english_path).dropna()

# Define English stopwords
stop_en = set(stopwords.words('english'))

def clean_text_en(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [w for w in tokens if w not in stop_en and len(w) > 1]
    return tokens

# Apply cleaning
df_en['cleaned_tokens'] = df_en['Review Text'].apply(clean_text_en)

# Show sample output
print(df_en[['Review Text', 'cleaned_tokens']].head(5))

# Save cleaned English data
df_en.to_csv('/Users/apple/Desktop/KD04403-NLP/ASSIGN/english_cleaned.csv', index=False)
print(" English cleaned dataset saved as english_cleaned.csv")


# -------------------------------------------------------------------
# SECTION B: BRAZILIAN PORTUGUESE DATASET
# -------------------------------------------------------------------
print("\n PREPROCESSING BRAZILIAN DATASET...")

# Load the Brazilian sample
brazil_path = '/Users/apple/Desktop/KD04403-NLP/ASSIGN/brazilian_sample.csv'
df_pt = pd.read_csv(brazil_path).dropna()

# Define Portuguese stopwords
stop_pt = set(stopwords.words('portuguese'))

def clean_text_pt(text):
    # Lowercase
    text = text.lower()
    # Keep only letters (including accented Portuguese chars)
    text = re.sub(r'[^a-záéíóúâêîôûãõç\s]', '', text)
    # Tokenize using Stanza
    doc = nlp_pt(text)
    tokens = [w.text for s in doc.sentences for w in s.words]
    # Remove stopwords
    tokens = [w for w in tokens if w not in stop_pt and len(w) > 1]
    return tokens

# Apply cleaning
df_pt['cleaned_tokens'] = df_pt['review_comment_message'].apply(clean_text_pt)

# Show sample output
print(df_pt[['review_comment_message', 'cleaned_tokens']].head(5))

# Save cleaned Portuguese data
df_pt.to_csv('/Users/apple/Desktop/KD04403-NLP/ASSIGN/brazilian_cleaned.csv', index=False)
print(" Brazilian cleaned dataset saved as brazilian_cleaned.csv")

# -------------------------------------------------------------------
# END OF TASK 1
# -------------------------------------------------------------------
print("\n Task 1 completed successfully! Cleaned datasets are ready.")
