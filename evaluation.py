import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import os

def evaluate_and_visualize(y_test, ngram_results, transformer_results, label_map):
    print("="*60)
    print("STEP 4: EVALUATION AND VISUALIZATION")
    print("="*60)
    
    y_test_indices = [label_map[label] for label in y_test]
    
    results = {
        'Model': ['1-gram', '2-gram', '3-gram', '4-gram', 'DistilBERT'],
        'Accuracy': [
            ngram_results['1-gram']['accuracy'],
            ngram_results['2-gram']['accuracy'],
            ngram_results['3-gram']['accuracy'],
            ngram_results['4-gram']['accuracy'],
            transformer_results['accuracy']
        ],
        'Precision': [
            ngram_results['1-gram']['precision'],
            ngram_results['2-gram']['precision'],
            ngram_results['3-gram']['precision'],
            ngram_results['4-gram']['precision'],
            transformer_results['precision']
        ],
        'Recall': [
            ngram_results['1-gram']['recall'],
            ngram_results['2-gram']['recall'],
            ngram_results['3-gram']['recall'],
            ngram_results['4-gram']['recall'],
            transformer_results['recall']
        ],
        'F1-Score': [
            ngram_results['1-gram']['f1'],
            ngram_results['2-gram']['f1'],
            ngram_results['3-gram']['f1'],
            ngram_results['4-gram']['f1'],
            transformer_results['f1']
        ]
    }
    results_df = pd.DataFrame(results)
    print("Performance Metrics:")
    print(results_df)
    
    os.makedirs('./results', exist_ok=True)
    results_df.to_csv('./results/performance_metrics.csv', index=False)
    
    plt.figure(figsize=(12, 6))
    results_melted = results_df.melt(id_vars='Model', value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'], 
                                    var_name='Metric', value_name='Score')
    sns.barplot(x='Model', y='Score', hue='Metric', data=results_melted)
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1)
    plt.savefig('./results/performance_barplot.png')
    plt.close()
    
    labels = list(label_map.keys())
    plt.figure(figsize=(15, 10))
    
    for i, model in enumerate(['1-gram', '2-gram', '3-gram', '4-gram', 'DistilBERT'], 1):
        plt.subplot(2, 3, i)
        if model == 'DistilBERT':
            cm = confusion_matrix(y_test_indices, transformer_results['predictions'], labels=list(label_map.values()))
        else:
            cm = confusion_matrix(y_test_indices, [label_map[p] for p in ngram_results[model]['predictions']], 
                                 labels=list(label_map.values()))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f'{model} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig('./results/confusion_matrices.png')
    plt.close()
    
    print("Visualizations saved to ./results/")