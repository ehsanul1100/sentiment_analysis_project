from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_and_test_transformer(X_train, X_test, y_train, y_test, model_name="DistilBERT"):
    print(f"Training {model_name} model...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    label_map = {label: idx for idx, label in enumerate(y_train.unique())}
    train_df = pd.DataFrame({'text': X_train, 'label': y_train.map(label_map)})
    test_df = pd.DataFrame({'text': X_test, 'label': y_test.map(label_map)})
    
    train_dataset = Dataset.from_pandas(train_df).map(
        lambda x: tokenizer(x['text'], padding="max_length", truncation=True, max_length=128), batched=True
    )
    test_dataset = Dataset.from_pandas(test_df).map(
        lambda x: tokenizer(x['text'], padding="max_length", truncation=True, max_length=128), batched=True
    )
    
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label_map))
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        no_cuda=False
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, average='weighted'),
            'recall': recall_score(labels, predictions, average='weighted'),
            'f1': f1_score(labels, predictions, average='weighted'),
            'predictions': predictions
        }
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()
    eval_results = trainer.evaluate()
    
    trainer.save_model('./results')
    tokenizer.save_pretrained('./results')
    
    return eval_results