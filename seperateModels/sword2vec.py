from fastapi import FastAPI, HTTPException, Query
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load
import numpy as np
from typing import List
from models import Item, SentenceCategory

app = FastAPI()

# Global variables to store model and Word2Vec
word2vec_model = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
classifier = LogisticRegression()

# Load model if available
try:
    word2vec_model = Word2Vec.load("word2vec_model.model")
    classifier = load("logistic_regression.joblib")
except Exception as e:
    print(f"Error loading model: {e}")

def sentence_to_avg_vector(sentence):
    words = sentence.split()
    word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    if not word_vectors:
        return np.zeros(word2vec_model.vector_size)
    return np.mean(word_vectors, axis=0)

@app.post("/train/")
async def train_model(train_data: List[SentenceCategory], update_mode: str = Query(..., description="Specify 'append' or 'replace' for data update mode."), test_size: float = 0.2):
    global word2vec_model, classifier

    sentences = [item.sentence for item in train_data]
    labels = [item.category for item in train_data]

    # Update Word2Vec model
    word2vec_model.build_vocab(sentences, update=True)
    word2vec_model.train(sentences, total_examples=len(sentences), epochs=10)

    # Prepare dataset for classifier
    X = np.array([sentence_to_avg_vector(sentence) for sentence in sentences])
    y = np.array(labels)

    # Train classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save models
    word2vec_model.save("word2vec_model.model")
    dump(classifier, "logistic_regression.joblib")

    return {"message": "Model trained successfully", "accuracy": accuracy}

@app.post("/predict/")
async def predict(item: Item):
    if not hasattr(classifier, 'coef_'):
        raise HTTPException(status_code=500, detail="Model is not fitted yet.")

    try:
        text_feature = sentence_to_avg_vector(item.text)
        prediction = classifier.predict([text_feature])
        return {"sentence": item.text, "prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
