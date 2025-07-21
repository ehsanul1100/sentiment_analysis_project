from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time

def train_and_test_ngram(X_train, X_test, y_train, y_test, ngram_range, model_name):
    """Train and test N-gram model"""
    print(f"\n--- Training and Testing {model_name} ---")
    
    start_time = time.time()
    
    # Create features
    print("  Creating features...")
    vectorizer = CountVectorizer(
        ngram_range=ngram_range, 
        max_features=5000,
        stop_words='english',
        lowercase=True
    )
    
    # Transform data
    print("  Transforming data...")
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    # Train model
    print("  Training model...")
    model = MultinomialNB()
    model.fit(X_train_vectorized, y_train)
    
    # Make predictions
    print("  Making predictions...")
    y_pred = model.predict(X_test_vectorized)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    end_time = time.time()
    
    print(f"  ✓ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  ✓ Precision: {precision:.4f}")
    print(f"  ✓ Recall: {recall:.4f}")
    print(f"  ✓ F1-Score: {f1:.4f}")
    print(f"  ✓ Training time: {end_time - start_time:.2f} seconds")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'time': end_time - start_time
    }

def test_all_ngram_models(X_train, X_test, y_train, y_test):
    """Test all N-gram models"""
    print("="*60)
    print("STEP 2: TRAINING AND TESTING N-GRAM MODELS")
    print("="*60)
    
    results = {}
    for n in range(1, 5):
        model_name = f"{n}-gram"
        results[model_name] = train_and_test_ngram(
            X_train, X_test, y_train, y_test, (n, n), f"{n}-gram"
        )
    
    print("\nN-gram models training completed!\n")
    return results