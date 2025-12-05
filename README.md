# Comparing-NLP-Techniques-on-English-vs-Portuguese-E-Commerce-Reviews
This project compares Natural Language Processing (NLP) techniques applied to two languages — English and Brazilian Portuguese — using e-commerce customer review datasets. The focus is on text preprocessing, Bigram Language Modelling with Laplace Smoothing, and Part-of-Speech (POS) Tagging, followed by accuracy evaluation and error analysis.


# Project Overview
This project evaluates how NLP models perform across different languages using real-world review data.

The main tasks include:

•Cleaning and preprocessing multilingual text

•Building unigram & bigram frequency models

•Applying Laplace smoothing

•Generating sentences based on trained language models

•Performing POS tagging using NLTK (English) and Stanza (Portuguese)

•Measuring tagging accuracy and analyzing frequent tagging errors

# Datasets

1. Women’s E-Commerce Clothing Reviews (English)

•Source: Kaggle

•Contains: 23,486 reviews

•Selected: 1,000 reviews

•Fields include: review text, rating, recommendation, etc.

Used as the English benchmark dataset.

3. Brazilian E-Commerce Public Dataset by Olist (Portuguese)
   
•Source: Kaggle

•Contains: 100,000+ review records

•Selected: 1,000 review entries

•Includes reviewer comments written in Brazilian Portuguese.

Used for multilingual comparison.

# Task 1: Text Pre-processing

English:

•Lowercase

•Remove punctuation & numbers

•Tokenize (NLTK)

•Remove stopwords

Portuguese:

•Tokenize (Stanza)

•Keep accents

•Remove Portuguese stopwords

Outputs:
english_cleaned.csv, brazilian_cleaned.csv

# Task 2: Bigram Language Model

•Built unigram + bigram counts

•Applied Laplace (Add-One) smoothing

•Generated 5 sentences for each language

Output: shows common review patterns like clothing fit (English) and delivery/defects (Portuguese).

# Task 3: POS Tagging

•English: NLTK pos_tag()
•Portuguese: Stanza Bosque model

Outputs:english_pos_tagged.csv, brazilian_pos_tagged.csv

#  Tools Used
•Python
•NLTK
•Stanza
•Pandas
•Regex

# Credits

This project was completed for the course: KD04403 – Natural Language Processing (NLP)
