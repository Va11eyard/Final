import numpy as np


def calculate_metrics(confusion_matrix):
    # Convert to numpy array for easier calculations
    cm = np.array(confusion_matrix)

    # Calculate accuracy
    accuracy = np.trace(cm) / np.sum(cm)

    # Calculate precision, recall, and F1-score for each class
    n_classes = cm.shape[0]
    metrics = {}

    for i in range(n_classes):
        # Precision
        precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() != 0 else 0

        # Recall
        recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() != 0 else 0

        # F1-score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        metrics[chr(97 + i)] = {
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1': round(f1, 3)
        }

    return round(accuracy, 3), metrics


# Define the confusion matrix from the table
confusion_matrix = [
    [30, 20, 10],
    [50, 60, 10],
    [20, 20, 80]
]

# Calculate metrics
accuracy, class_metrics = calculate_metrics(confusion_matrix)

print(f"a) Accuracy: {accuracy}")
print("\nb) Precision and Recall for each class:")
for class_name, metrics in class_metrics.items():
    print(f"\nClass {class_name}:")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1-score: {metrics['f1']}")

print("\nMetrics table format:")
print("\nClass | Precision | Recall | F1-score")
print("-" * 35)
for class_name, metrics in class_metrics.items():
    print(f"{class_name}     {metrics['precision']}     {metrics['recall']}    {metrics['f1']}")

'''
a) Accuracy: 0.567

b) Precision and Recall for each class:

Metrics table format:

Class | Precision | Recall | F1-score
-----------------------------------
a     0.3     0.5    0.375
b     0.6     0.5    0.545
c     0.8     0.667    0.727


'''