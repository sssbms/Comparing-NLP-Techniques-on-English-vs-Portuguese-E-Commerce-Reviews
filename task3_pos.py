# task3_pos_tagging.py

import pandas as pd
import nltk
import stanza
import ast

# Make sure NLTK POS tagger data is available
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
    
# Initialize Stanza pipeline for Portuguese
stanza.download('pt')
nlp_pt = stanza.Pipeline(lang='pt', processors='tokenize,pos', use_gpu=False)

# ============================================================ #
# Load Cleaned Datasets
# ============================================================ #
english_df = pd.read_csv("/Users/apple/Desktop/KD04403-NLP/ASSIGN/english_cleaned.csv")
brazilian_df = pd.read_csv("/Users/apple/Desktop/KD04403-NLP/ASSIGN/brazilian_cleaned.csv")


english_df["cleaned_tokens"] = english_df["cleaned_tokens"].apply(ast.literal_eval)
brazilian_df["cleaned_tokens"] = brazilian_df["cleaned_tokens"].apply(ast.literal_eval)

# ============================================================ #
# POS Tagging (English using NLTK)
# ============================================================ #
def pos_tag_english(tokens_list):
    tagged_sentences = []
    for tokens in tokens_list:
        tagged = nltk.pos_tag(tokens)
        tagged_sentences.append(tagged)
    return tagged_sentences

print("🇬🇧 Tagging English dataset...")
english_df["pos_tags"] = pos_tag_english(english_df["cleaned_tokens"])
english_df.to_csv("english_pos_tagged.csv", index=False)
print("English POS tagging completed and saved to english_pos_tagged.csv")

# ============================================================ #
#  POS Tagging (Portuguese using Stanza)
# ============================================================ #
def pos_tag_portuguese(tokens_list):
    tagged_sentences = []
    for tokens in tokens_list:
        text = " ".join(tokens)
        doc = nlp_pt(text)
        tagged = [(w.text, w.upos) for s in doc.sentences for w in s.words]
        tagged_sentences.append(tagged)
    return tagged_sentences

print(" Tagging Portuguese dataset...")
brazilian_df["pos_tags"] = pos_tag_portuguese(brazilian_df["cleaned_tokens"])
brazilian_df.to_csv("brazilian_pos_tagged.csv", index=False)
print(" Portuguese POS tagging completed and saved to brazilian_pos_tagged.csv")

# ============================================================ #
#  Display a few tagged samples for your report
# ============================================================ #
print("\n--- Sample English POS Tags ---")
for row in english_df["pos_tags"].head(3):
    print(row, "\n")

print("\n--- Sample Portuguese POS Tags ---")
for row in brazilian_df["pos_tags"].head(3):
    print(row, "\n")

# ============================================================ #
#  Simple Manual Accuracy Test Placeholder
# ============================================================ #
sample_english = english_df.head(5)[["cleaned_tokens", "pos_tags"]]
sample_portuguese = brazilian_df.head(5)[["cleaned_tokens", "pos_tags"]]

sample_english.to_csv("english_pos_sample_for_check.csv", index=False)
sample_portuguese.to_csv("brazilian_pos_sample_for_check.csv", index=False)

print("\n Saved small samples for manual accuracy checking:")
print("   english_pos_sample_for_check.csv")
print("   brazilian_pos_sample_for_check.csv")

print("\n Task 3 completed successfully! POS tags generated and saved.")
