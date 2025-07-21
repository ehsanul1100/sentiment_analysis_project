import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(dataset_path=None, sample_size=50000, use_sst5=False):
    """Load and prepare dataset (custom or SST-5)"""
    print("="*60)
    print("STEP 1: LOADING AND PREPARING DATA")
    print("="*60)
    
    if use_sst5:
        print("Loading SST-5 dataset...")
        dataset = load_dataset("sst")
        df = pd.DataFrame({
            'review': dataset['train']['sentence'],
            'sentiment': dataset['train']['label']
        })
        # Map SST-5 labels (0-1 continuous) to 5 classes
        df['sentiment'] = pd.cut(df['sentiment'], bins=5, labels=[
            'very_negative', 'negative', 'neutral', 'positive', 'very_positive'
        ])
    else:
        print(f"Loading custom dataset from {dataset_path}...")
        df = pd.read_csv(dataset_path)
    
    print(f"Original dataset size: {len(df)} reviews")
    
    # Display class distribution
    print("Class distribution:")
    print(df['sentiment'].value_counts())
    
    # Take balanced sample if specified
    if sample_size and len(df) > sample_size:
        print(f"Taking sample of {sample_size} reviews...")
        classes = df['sentiment'].unique()
        sample_dfs = []
        samples_per_class = sample_size // len(classes)
        
        for cls in classes:
            class_df = df[df['sentiment'] == cls].sample(samples_per_class)
            sample_dfs.append(class_df)
        
        df = pd.concat(sample_dfs).sample(frac=1).reset_index(drop=True)
        print(f"Sample dataset size: {len(df)} reviews")
    
    # Split into train and test sets
    X = df['review']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} reviews")
    print(f"Test set: {len(X_test)} reviews")
    print("Data preparation completed!\n")
    
    return X_train, X_test, y_train, y_test