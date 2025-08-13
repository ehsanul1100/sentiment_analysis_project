import data_preparation
import ngram_models
import transformer_models
import evaluation
import os
import pandas as pd

def main():
    print("SENTIMENT ANALYSIS MODEL COMPARISON")
    print("Training and Testing N-gram (1,2,3,4) vs Transformer Models")
    print("="*80)

    # Use local SST-5 CSV files
    TRAIN_FILE = "sst_5_train.csv"
    TEST_FILE = "sst_5_test.csv"
    SAMPLE_SIZE = None
    USE_SST5 = False  # We'll load our own CSVs

    # Load train and test CSVs and concatenate for unified processing
    X_train, X_test, y_train, y_test = data_preparation.load_and_prepare_data(
        train_path=TRAIN_FILE,
        test_path=TEST_FILE,
        sample_size=SAMPLE_SIZE
    )

    print(f"Training set: {len(X_train)} reviews")
    print(f"Test set: {len(X_test)} reviews")

    print("="*60)
    print("STEP 2: TRAINING N-GRAM MODELS")
    print("="*60)
    ngram_results = {}
    for n in [1, 2, 3, 4]:
        ngram_results[f"{n}-gram"] = ngram_models.train_and_test_ngram(
            X_train, X_test, y_train, y_test,
            ngram_range=(1, n), model_name=f"{n}-gram"
        )

    print("="*60)
    print("STEP 3: TRAINING TRANSFORMER MODEL")
    print("="*60)
    distilbert_results = transformer_models.train_and_test_transformer(
        X_train, X_test, y_train, y_test
    )

    label_map = {label: idx for idx, label in enumerate(pd.concat([y_train, y_test]).unique())}
    evaluation.evaluate_and_visualize(y_test, ngram_results, distilbert_results, label_map)

    # Ensure all results are in ./results (no Google Drive copy)
    print("All results saved in ./results directory.")

if __name__ == "__main__":
    main()