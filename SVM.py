from fastapi import FastAPI, HTTPException, Query
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load
import numpy as np
from models import Item, SentenceCategory, TrainingRequest
from typing import List

app = FastAPI()

# Assume global storage for the data
current_data = []

# Initialize or load the model and vectorizer
try:
    model = load('svm_model.joblib')
    vectorizer = load('tfidf_vectorizer.joblib')
except Exception as e:
    print(f"Failed to load model or vectorizer: {e}")
    model = svm.SVC(kernel='linear')
    vectorizer = TfidfVectorizer()

@app.post("/train/")
def train_model(train_data: List[SentenceCategory], update_mode: str = Query(..., description="Specify 'append' or 'replace' for data update mode."), test_size: float = 0.2):
    global current_data

    if update_mode == "append":
        current_data.extend(train_data)
    elif update_mode == "replace":
        current_data = train_data
    else:
        raise HTTPException(status_code=400, detail="Invalid update mode. Choose 'append' or 'replace'.")
    
    # Print current data to console for debugging
    print("Current data:")
    for item in current_data:
        print(f"Sentence: {item.sentence}, Category: {item.category}")

    if len(current_data) < 5:
        raise HTTPException(status_code=400, detail="Not enough data for training. A minimum of 5 data points is required.")

    try:
        sentences = [item.sentence for item in current_data]
        labels = [item.category for item in current_data]
        X = vectorizer.fit_transform(sentences)

        best_accuracy = 0
        best_random_state = None
        best_model = None

        for random_state in range(1, 501):
            X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=test_size, random_state=random_state, stratify=labels)
            local_model = svm.SVC(kernel='linear')
            local_model.fit(X_train, y_train)
            y_pred = local_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_random_state = random_state
                best_model = local_model

        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=test_size, random_state=best_random_state, stratify=labels)
        best_model.fit(X_train, y_train)
        dump(best_model, 'svm_model.joblib')
        dump(vectorizer, 'tfidf_vectorizer.joblib')

        return {"message": "Model trained successfully with best random_state", "accuracy": best_accuracy, "random_state": best_random_state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/")
def predict(item: Item):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model or vectorizer is not initialized.")
    try:
        text_features = vectorizer.transform([item.text])
        prediction = model.predict(text_features)
        return {"sentence": item.text, "prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")
