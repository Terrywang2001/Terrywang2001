from fastapi import FastAPI, HTTPException, Query
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from joblib import dump, load
import numpy as np
from typing import List
from models import Item, SentenceCategory
import os


app = FastAPI()


# Global variables to store models and vectorizers
tfidf_vectorizer = TfidfVectorizer()
svm_model = SVC(kernel='linear')
logistic_model = LogisticRegression()
count_vectorizer = CountVectorizer()
naive_bayes_model = MultinomialNB()

# Global training data list
current_data = []

# Load model and vectorizer if available
def load_models_and_vectorizers():
    global tfidf_vectorizer, svm_model, logistic_model, count_vectorizer, naive_bayes_model
    try:
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib') if os.path.exists('tfidf_vectorizer.joblib') else tfidf_vectorizer
        svm_model = joblib.load('svm_model.joblib') if os.path.exists('svm_model.joblib') else svm_model
        logistic_model = joblib.load('logistic_regression.joblib') if os.path.exists('logistic_regression.joblib') else logistic_model
        count_vectorizer = joblib.load('count_vectorizer.joblib') if os.path.exists('count_vectorizer.joblib') else count_vectorizer
        naive_bayes_model = joblib.load('naive_bayes_model.joblib') if os.path.exists('naive_bayes_model.joblib') else naive_bayes_model
    except Exception as e:
        print(f"Error loading models or vectorizers: {e}")

load_models_and_vectorizers()

@app.post("/train/")
async def train_model(
    train_data: List[SentenceCategory],
    model_type: str = Query(..., description="Specify 'svm' or 'naive_bayes' for model type."),
    update_mode: str = Query(..., description="Specify 'append' or 'replace' for data update mode."),
    test_size: float = 0.2
):
    global current_data, tfidf_vectorizer, svm_model, logistic_model, count_vectorizer, naive_bayes_model
    
    # Debug print statements
    print(f"Received model_type: {model_type}")
    print(f"Received update_mode: {update_mode}")
    print(f"Received train_data: {train_data}")

    # Update or replace training data based on user input
    if update_mode == "append":
        current_data.extend(train_data)
    elif update_mode == "replace":
        current_data = train_data
    else:
        raise HTTPException(status_code=400, detail="Invalid update mode specified.")

    best_accuracy = 0
    best_random_state = None
    best_model = None
    best_vectorizer = None

    sentences = [item.sentence for item in current_data]
    labels = [item.category for item in current_data]

    if model_type == "svm":
        # Training logic for SVM + TF-IDF
        X = tfidf_vectorizer.fit_transform(sentences)
        y = labels

        for random_state in range(1, 501):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
            local_model = SVC(kernel='linear')
            local_model.fit(X_train, y_train)
            y_pred = local_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_random_state = random_state
                best_model = local_model
                best_vectorizer = tfidf_vectorizer
        
        svm_model = best_model
        joblib.dump(svm_model, 'svm_model.joblib')
        joblib.dump(best_vectorizer, 'tfidf_vectorizer.joblib')

    elif model_type == "naive_bayes":
        # Training logic for Naive Bayes using CountVectorizer
        X = count_vectorizer.fit_transform(sentences)
        y = np.array(labels)

        for random_state in range(1, 501):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            local_model = MultinomialNB()
            local_model.fit(X_train, y_train)
            y_pred = local_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_random_state = random_state
                best_model = local_model
                best_vectorizer = count_vectorizer
        
        naive_bayes_model = best_model
        joblib.dump(naive_bayes_model, 'naive_bayes_model.joblib')
        joblib.dump(best_vectorizer, 'count_vectorizer.joblib')
    else:
        raise HTTPException(status_code=400, detail="Invalid model type specified.")

    return {"message": f"Model trained successfully using {model_type}", "accuracy": best_accuracy, "random_state": best_random_state}

@app.post("/predict/")
async def predict(item: Item, model_type: str = Query(..., description="Specify 'svm' or 'naive_bayes' for model type.")):
    global tfidf_vectorizer, svm_model, logistic_model, count_vectorizer, naive_bayes_model

    if model_type == "svm":
        if not hasattr(svm_model, "support_"):
            raise HTTPException(status_code=500, detail="SVM model is not fitted yet.")
        text_features = tfidf_vectorizer.transform([item.text])
        prediction = svm_model.predict(text_features)
    elif model_type == "naive_bayes":
        if not hasattr(naive_bayes_model, 'class_count_'):
            raise HTTPException(status_code=500, detail="Naive Bayes model is not fitted yet.")
        text_features = count_vectorizer.transform([item.text])
        prediction = naive_bayes_model.predict(text_features)
    else:
        raise HTTPException(status_code=400, detail="Invalid model type specified.")

    return {"sentence": item.text, "prediction": prediction[0]}

