from sklearn.metrics import confusion_matrix
import DATA221_Assignment4_Question3 as q3
import DATA221_Assignment4_Question4 as q4

decision_tree_confusion = confusion_matrix(q3.y_test, q3.y_test_prediction) # build confusion matrix from decision tree in question 3

neural_networks_probability_prediction = q4.neural_network_model.predict(q4.X_test_scaled) # get question 4 sigmoid probabilities
neural_networks_predictions = (neural_networks_probability_prediction[:, 0] >= 0.5).astype(int) # convert probabilities to class labels
neural_networks_confusion = confusion_matrix(q4.y_test, neural_networks_predictions) # build confusion matrix for neural network in question 4

print('Decision Tree confusion matrix: ')
print(decision_tree_confusion)
print('Neural Network confusion matrix: ')
print(neural_networks_confusion)

# I would prefer the constrained Decision Tree for this task
# Decision Tree advantage: easy to explain
# Decision Tree limitation: can miss complex patterns
# Neural Network advantage: could learn more complex patterns
# Neural Network limitation: hard to explain
