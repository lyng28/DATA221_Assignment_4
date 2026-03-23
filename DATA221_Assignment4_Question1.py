from sklearn.datasets import load_breast_cancer
from collections import Counter

breast_cancer_dataset = load_breast_cancer()
X = breast_cancer_dataset.data
y = breast_cancer_dataset.target

print("X shape: ", X.shape)
print("y shape: ", y.shape)

class_counts = Counter(y)
for class_id in sorted(class_counts):
    class_count = class_counts[class_id]
    class_name = breast_cancer_dataset.target_names[class_id]

    print(f'Class {class_id} ({class_name}): {class_count}')

# The data set is slightly imbalanced because there are more class 1 (benign) than class 0 (malignant)
# Class balance is an important consideration for classification models because a model can appear to be accurate by predicting
# the majority class too often