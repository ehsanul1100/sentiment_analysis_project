from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

def train_and_test_ngram(X_train, X_test, y_train, y_test, ngram_range=(1,1), model_name="1-gram"):
    print(f"Training {model_name} model...")
    vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=1000, stop_words='english')
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    model = MultinomialNB()
    model.fit(X_train_vectorized, y_train)
    predictions = model.predict(X_test_vectorized)
    
    os.makedirs('./results', exist_ok=True)
    joblib.dump(model, f'./results/{model_name}_model.pkl')
    joblib.dump(vectorizer, f'./results/{model_name}_vectorizer.pkl')
    
    return {
        'predictions': predictions,
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions, average='weighted'),
        'recall': recall_score(y_test, predictions, average='weighted'),
        'f1': f1_score(y_test, predictions, average='weighted')
    }