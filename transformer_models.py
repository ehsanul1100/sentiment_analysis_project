from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
from datasets import Dataset

def train_and_test_transformer(X_train, X_test, y_train, y_test, model_name="DistilBERT"):
    """Train and test DistilBERT model"""
    print(f"\n--- Training and Testing {model_name} ---")
    
    start_time = time.time()
    
    # Prepare dataset
    print("  Preparing dataset...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Map labels to integers
    label_map = {label: idx for idx, label in enumerate(y_train.unique())}
    y_train_int = y_train.map(label_map)
    y_test_int = y_test.map(label_map)
    
    # Create datasets
    train_df = pd.DataFrame({'text': X_train, 'label': y_train_int})
    test_df = pd.DataFrame({'text': X_test, 'label': y_test_int})
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Initialize model
    print("  Initializing model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=len(label_map)
    )
    
    # Training arguments optimized for CPU
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        no_cuda=True  # Force CPU usage
    )
    
    # Define metrics
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train model
    print("  Training model...")
    trainer.train()
    
    # Evaluate model
    print("  Evaluating model...")
    eval_results = trainer.evaluate()
    
    end_time = time.time()
    
    print(f"  ✓ Accuracy: {eval_results['eval_accuracy']:.4f} ({eval_results['eval_accuracy']*100:.2f}%)")
    print(f"  ✓ Precision: {eval_results['eval_precision']:.4f}")
    print(f"  ✓ Recall: {eval_results['eval_recall']:.4f}")
    print(f"  ✓ F1-Score: {eval_results['eval_f1']:.4f}")
    print(f"  ✓ Training time: {end_time - start_time:.2f} seconds")
    
    return {
        'accuracy': eval_results['eval_accuracy'],
        'precision': eval_results['eval_precision'],
        'recall': eval_results['eval_recall'],
        'f1': eval_results['eval_f1'],
        'time': end_time - start_time
    }

def test_transformer_models(X_train, X_test, y_train, y_test):
    """Test transformer models"""
    print("="*60)
    print("STEP 3: TRAINING AND TESTING TRANSFORMER MODELS")
    print("="*60)
    
    results = {}
    results['DistilBERT'] = train_and_test_transformer(X_train, X_test, y_train, y_test)
    
    print("\nTransformer models training completed!\n")
    return results