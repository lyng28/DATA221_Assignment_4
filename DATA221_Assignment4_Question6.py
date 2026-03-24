import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

cnn_model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(units=64, activation='relu'),
    layers.Dropout(rate=0.3),
    layers.Dense(units=10, activation='softmax'),
])

cnn_model.summary() # Print model layers and parameter counts

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

training_history = cnn_model.fit(X_train, y_train, validation_split=0.1, epochs=15, batch_size=64)

test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test)

print('Test loss: ', test_loss)
print('Test accuracy: ', test_accuracy)

# CNNs are preferred for images because they learn local patterns and use fewer parameters
# In this task the convolution layers learn useful clothing features such as edges, textures, and shape parts to help classify each item

