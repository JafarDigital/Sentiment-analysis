import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def collect_data():
    data = []
    print("Enter text data (or type 'stop' to end):")
    while True:
        review = input("> ")
        if review.lower() == 'stop':
            break
        data.append(review)
    return data

def clean_text(text):
    text = text.lower()  # Convert all text to lowercase
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation and symbols
    return text

# Extract unique words (generating the vocab)
def generate_vocab(data):
    cleaned_data = [clean_text(review) for review in data]
    return list(set(" ".join(cleaned_data).split()))  # Returns a list of unique words by using sets

# Generate embeddings using TF-IDF
def generate_embeddings(data, vocab):
    vectorizer = TfidfVectorizer(vocabulary=vocab)  # Create a TF-IDF vectorizer with the vocabulary
    tfidf_matrix = vectorizer.fit_transform(data)  # Transform the data into TF-IDF values
    
    # Extract the TF-IDF values for each word in the vocab
    embeddings = tfidf_matrix.toarray()
    
    return embeddings

def save_results(vocab, embeddings, vocab_file, embeddings_file):
    with open(vocab_file, 'w') as f:
        for word in vocab:
            f.write(f"{word}\n")

    np.savetxt(embeddings_file, embeddings, delimiter=',')
    print(f"vocab and embeddings files are saved in {vocab_file} and {embeddings_file}")

def main():
    data = collect_data() 
    vocab = generate_vocab(data)
    embeddings = generate_embeddings(data, vocab)
    
    save_results(vocab, embeddings, 'custom_vocab.txt', 'custom_embeddings.txt')

if __name__ == "__main__":
    main() 
