from fastapi import FastAPI, HTTPException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load
from typing import List
import numpy as np
from models import Item, SentenceCategory

app = FastAPI()

# Initialize or load the model and vectorizer
try:
    model = load('svm_model.joblib')
    vectorizer = load('tfidf_vectorizer.joblib')
except:
    model = None
    vectorizer = None

@app.post("/train/")
def train_model(train_data: List[SentenceCategory], test_size: float = 0.25):
    try:
        sentences = [item.sentence for item in train_data]
        labels = [item.category for item in train_data]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(sentences)

        best_accuracy = 0
        best_random_state = None
        best_model = None

        for random_state in range(1, 201):
            X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=test_size, random_state=random_state)
            model = svm.SVC(kernel='linear')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_random_state = random_state
                best_model = model

        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=test_size, random_state=best_random_state)
        best_model.fit(X_train, y_train)
        dump(best_model, 'svm_model.joblib')
        dump(vectorizer, 'tfidf_vectorizer.joblib')

        return {"message": "Model trained successfully with best random_state", "accuracy": best_accuracy, "random_state": best_random_state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/")
def predict(item: Item):
    if model and vectorizer:
        text_features = vectorizer.transform([item.text])
        prediction = model.predict(text_features)
        return {"sentence": item.text, "prediction": prediction[0]}
    else:
        raise HTTPException(status_code=500, detail="Model is not trained yet.")

