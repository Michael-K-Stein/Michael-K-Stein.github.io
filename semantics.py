#pip install nltk
#pip install -U scikit-learn

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def semantic_similarity(paragraph1, paragraph2):
    # Preprocess text
    stop_words = set(stopwords.words('english'))
    paragraph1 = ' '.join([word.lower() for word in nltk.word_tokenize(paragraph1) if word.lower() not in stop_words])
    paragraph2 = ' '.join([word.lower() for word in nltk.word_tokenize(paragraph2) if word.lower() not in stop_words])

    # Compute similarity score
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([paragraph1, paragraph2])
    similarity_score = cosine_similarity(tfidf_matrix)[0][1]

    return similarity_score
