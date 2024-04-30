from fastapi import FastAPI, HTTPException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load
import numpy as np
from models import Item, SentenceCategoryList

app = FastAPI()

# Initialize or load the model and vectorizer
try:
    model = load('svm_model.joblib')
    vectorizer = load('tfidf_vectorizer.joblib')
except:
    model = None
    vectorizer = None

@app.post("/train/")
def train_model(train_data: SentenceCategoryList, test_size: float = 0.25, random_state: int = 42):
    sentences = [data.sentence for data in train_data.data]
    labels = [data.category for data in train_data.data]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=test_size, random_state=random_state)
    model = svm.SVC(kernel='linear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    dump(model, 'svm_model.joblib')
    dump(vectorizer, 'tfidf_vectorizer.joblib')
    return {"message": "Model trained successfully", "accuracy": accuracy}

@app.post("/predict/")
def predict(item: Item):
    if model and vectorizer:
        text_features = vectorizer.transform([item.text])
        prediction = model.predict(text_features)
        return {"sentence": item.text, "prediction": prediction[0]}
    else:
        raise HTTPException(status_code=500, detail="Model is not trained yet.")

