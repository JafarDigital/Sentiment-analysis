# predict.py

import os
import random
import re

# Loading the vocab dataset and its embeddings [if any]
DATA_DIR = "dataset"
VOCAB_PATH = os.path.join(DATA_DIR, "imdb.vocab")
EMB_PATH = os.path.join(DATA_DIR, "imdbEr.txt")

# We make a set of example movies
MOVIES = [
    "Star Wars", "Gladiator", "Lord of the Rings",
    "Person of Interest", "One Flew Over the Cuckoo's Nest", "The Matrix",
    "Dune", "Interstellar", "The Shawshank Redemption", "Inception"
]

def load_vocab_and_embeddings():
	# Reading the vocab dataset
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f]

    # Reading the embeddings dataset
    with open(EMB_PATH, "r", encoding="utf-8") as f:
        embeddings = [float(line.strip()) for line in f]

    # Make a dictionary to link the words with its embeddings
    word2score = {word: score for word, score in zip(vocab, embeddings)}
    return word2score

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Removing punctuation and symbols from the text
    return text.split()

def analyze_sentiment(review, word2score):
    words = clean_text(review)
    scores = [word2score[word] for word in words if word in word2score]

    if not scores:
        return "Unknown (no known words found)"

    avg = sum(scores) / len(scores) # Calculates the average sentiment score for all words in the user's review

    if avg > 0.05:
        return "Positive"
    elif avg < -0.05:
        return "Negative"
    else:
        return "Neutral"

def main():
    word2score = load_vocab_and_embeddings()

    movie = random.choice(MOVIES)
    print(f"\nSelected movie/series: **{movie}**")

    review = input("\nEnter your review:\n> ")

    sentiment = analyze_sentiment(review, word2score)

    print(f"\nSentiment: {sentiment}\n")

if __name__ == "__main__":
    main()
