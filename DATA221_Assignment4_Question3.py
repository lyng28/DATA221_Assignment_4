from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

breast_cancer_dataset = load_breast_cancer()
X = breast_cancer_dataset.data
y = breast_cancer_dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

constrained_tree_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=8, random_state=42)

constrained_tree_classifier.fit(X_train, y_train)

y_train_prediction = constrained_tree_classifier.predict(X_train)
y_test_prediction = constrained_tree_classifier.predict(X_test)

train_accuracy_score = accuracy_score(y_train, y_train_prediction)
test_accuracy_score = accuracy_score(y_test, y_test_prediction)

print('Train accuracy: ', train_accuracy_score)
print('Test accuracy: ', test_accuracy_score)

# learn code from https://stackoverflow.com/questions/23900080/how-are-feature-importances-ordered-in-scikit-learns-randomforestregressor
# build list of feature name and important score
important_features = list(zip(breast_cancer_dataset.feature_names, constrained_tree_classifier.feature_importances_))
important_features.sort(key=lambda pair: pair[1], reverse=True) # sort the list

print('Top five most important features:')
rank = 1
for feature_name, feature_score in important_features[:5]:  # loop through top five features.
    print(f'{rank}. {feature_name}: {feature_score:.4f}')  # print feature rank, name, and score.
    rank+= 1

# Controlling model complexity reduces overfitting by stopping tree from learning tiny specific patterns
# Feature importance supports interpretability because it shows which inputs had the strongest impact on decision

