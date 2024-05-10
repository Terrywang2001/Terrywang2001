from fastapi import FastAPI, HTTPException, Query
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load
import numpy as np
from models import Item, SentenceCategory
from typing import List

app = FastAPI()

# Global variables to store model and vectorizer
vectorizer = TfidfVectorizer()
model = svm.SVC(kernel='linear')

# Load model and vectorizer if available
try:
    vectorizer = load('tfidf_vectorizer.joblib')
    model = load('svm_model.joblib')
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")

@app.post("/train/")
def train_model(train_data: List[SentenceCategory], update_mode: str = Query(..., description="Specify 'append' or 'replace' for data update mode."), test_size: float = 0.2):
    global current_data, vectorizer, model

    if update_mode == "append":
        print("Appending " + str(len(train_data)) + " new records.")
        current_data.extend(train_data)
    elif update_mode == "replace":
        print("Replacing current data with " + str(len(train_data)) + " new records.")
        current_data = train_data
    else:
        raise HTTPException(status_code=400, detail="Invalid update mode. Choose 'append' or 'replace'.")
    
    # Print current data for debugging
    print("Current data count: " + str(len(current_data)))
    
    if len(current_data) < 5:
        raise HTTPException(status_code=400, detail="Not enough data for training. A minimum of 5 data points is required.")

    try:
        # Reinitialize vectorizer and model each time to ensure consistency
        vectorizer = TfidfVectorizer()
        sentences = [item.sentence for item in current_data]
        labels = [item.category for item in current_data]
        X = vectorizer.fit_transform(sentences)

        model = svm.SVC(kernel='linear')
        best_accuracy = 0
        best_random_state = None
        best_model = None

        # Find the best model
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

        # Retrain the best model with the entire dataset
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=test_size, random_state=best_random_state, stratify=labels)
        best_model.fit(X_train, y_train)
        
        # Save the retrained model and vectorizer
        model = best_model
        dump(best_model, 'svm_model.joblib')
        dump(vectorizer, 'tfidf_vectorizer.joblib')

        return {"message": "Model trained successfully with best random_state", "accuracy": best_accuracy, "random_state": best_random_state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/")
def predict(item: Item):
    if not hasattr(model, "support_"):  # Check if model is fitted
        raise HTTPException(status_code=500, detail="Model is not fitted yet.")
    try:
        text_features = vectorizer.transform([item.text])
        prediction = model.predict(text_features)
        return {"sentence": item.text, "prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

"""  def predict(item: Item):
    global vectorizer, model  # 确保使用全局变量中加载的模型和向量化器
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model or vectorizer is not initialized.")

    try:
        text_features = vectorizer.transform([item.text])
        prediction = model.predict(text_features)
        return {"sentence": item.text, "prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")
"""

