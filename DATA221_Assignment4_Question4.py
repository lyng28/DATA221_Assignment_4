from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer

breast_cancer_dataset = load_breast_cancer()
X = breast_cancer_dataset.data
y = breast_cancer_dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

tf.random.set_seed(1)
neural_network_model = Sequential()
input_layer = InputLayer(shape=(30,))
hidden_layer = Dense(16, activation="relu") # hidden layer with 16 neurons, use relu for model to also learn non-linear patterns
output_layer = Dense(1, activation="sigmoid")

neural_network_model.add(input_layer)
neural_network_model.add(hidden_layer)
neural_network_model.add(output_layer)

neural_network_model.compile(loss='binary_crossentropy', metrics=["accuracy"])
neural_network_model.fit(X_train_scaled, y_train, epochs=10)

train_loss, train_accuracy = neural_network_model.evaluate(X_train_scaled, y_train)  # Evaluate train accuracy.
test_loss, test_accuracy = neural_network_model.evaluate(X_test_scaled, y_test)  # Evaluate test accuracy.

print('Train accuracy: ', train_accuracy)
print('Test accuracy: ', test_accuracy)

# Feature scaling is needed because neural networks optimize with gradients and large feature scale differences hurt learning
# An epoch represents one full loop through all training samples during model training

