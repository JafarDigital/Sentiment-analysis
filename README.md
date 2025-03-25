# Sentiment Analysis with Custom Vocab and Embeddings

This project enables sentiment analysis using custom-generated vocabularies and word embeddings, as well as using a pre-existing dataset such as the IMDb reviews dataset. The goal is to train a model that can classify text data (like movie reviews) into positive or negative sentiment.

## Features
- **Custom Vocab and Embedding Generation**: Users can input their own text data to generate a custom vocabulary and corresponding embeddings using the **TF-IDF** method.
- **Sentiment Analysis on IMDb Dataset**: Uses the IMDb dataset, preprocessed and tokenized, for sentiment analysis, to classify reviews into positive or negative categories.
- **Model Training**: A simple model is built using **PyTorch** for sentiment classification based on the embeddings.
- **User Input for Predictions**: After training, the user can input their own review, and the model will predict whether it's positive or negative.

## Project Workflow

1. **Data Collection**:
   - Custom data: Users can provide their own dataset (reviews, comments, etc.).
   - IMDb dataset: A ready-to-use dataset containing movie reviews along with sentiment labels.
   
2. **Preprocessing**:
   - Tokenizes and cleans the text, removing unnecessary punctuation and converting to lowercase.
   
3. **Vocabulary & Embedding Generation**:
   - The program generates a vocabulary of unique words from the dataset.
   - Using **TF-IDF**, word embeddings are calculated to represent the importance of each word.

4. **Model Training**:
   - A simple PyTorch model is trained using the embeddings to classify the sentiment of reviews as positive or negative.
   
5. **Prediction**:
   - After the model is trained, users can input their own reviews to see the sentiment classification.

## To install the required libraries, run:

pip install torch scikit-learn numpy beautifulsoup4 requests
