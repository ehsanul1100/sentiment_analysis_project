import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(train_path, test_path, sample_size=None):
    print("="*60)
    print("STEP 1: LOADING AND PREPARING DATA")
    print("="*60)

    print(f"Loading custom train set from {train_path} and test set from {test_path}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    df = pd.concat([train_df, test_df]).reset_index(drop=True)
    df = df.dropna(subset=['sentiment', 'review'])
    print(f"Original dataset size: {len(df)} reviews")
    print("Class distribution:")
    print(df['sentiment'].value_counts())

    # Optionally sample
    if sample_size and len(df) > sample_size:
        print(f"Taking sample of {sample_size} reviews...")
        classes = df['sentiment'].unique()
        sample_dfs = []
        samples_per_class = sample_size // len(classes)
        for cls in classes:
            class_df = df[df['sentiment'] == cls].sample(samples_per_class, random_state=42)
            sample_dfs.append(class_df)
        df = pd.concat(sample_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Sample dataset size: {len(df)} reviews")

    # Use original splits for train/test
    X_train = train_df['review']
    y_train = train_df['sentiment']
    X_test = test_df['review']
    y_test = test_df['sentiment']

    print(f"Training set: {len(X_train)} reviews")
    print(f"Test set: {len(X_test)} reviews")
    print("Data preparation completed!\n")
    return X_train, X_test, y_train, y_test