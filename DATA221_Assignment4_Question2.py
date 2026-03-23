from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

breast_cancer_dataset = load_breast_cancer()
X = breast_cancer_dataset.data
y = breast_cancer_dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

decision_tree_classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)

decision_tree_classifier.fit(X_train, y_train)

y_train_prediction = decision_tree_classifier.predict(X_train)
y_test_prediction = decision_tree_classifier.predict(X_test)

train_accuracy_score = accuracy_score(y_train, y_train_prediction)
test_accuracy_score = accuracy_score(y_test, y_test_prediction)

print('Train accuracy: ', train_accuracy_score)
print('Test accuracy: ', test_accuracy_score)

# Entropy measure the uncertainty in a node, and the tree chooses splits that reduces this uncertainty
# The train and test accuracy in this case are good generalization as they are both high and quite close to one another