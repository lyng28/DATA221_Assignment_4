import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import DATA221_Assignment4_Question6 as q6

fashion_class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']   # label names for class ids 0 to 9, https://www.tensorflow.org/tutorials/keras/classification

predict_probabilities = q6.cnn_model.predict(q6.X_test, verbose=0)
predicted_labels = np.argmax(predict_probabilities, axis=1) # learn from https://www.geeksforgeeks.org/python/numpy-argmax-python/

cnn_confusion = confusion_matrix(q6.y_test, predicted_labels)
print('CNN confusion matrix: ', cnn_confusion)

misclassified_indices = np.where(predicted_labels != q6.y_test)[0]
selected_indices = misclassified_indices[:3]

figure, axes = plt.subplots(1, 3)
for plot_index in range(3):
    image_index = selected_indices[plot_index]
    true_name = fashion_class_names[q6.y_test[image_index]]
    predicted_name = fashion_class_names[predicted_labels[image_index]]
    axes[plot_index].imshow(q6.X_test[image_index], cmap='gray')
    axes[plot_index].set_title(f'True: {true_name}\nPredicted: {predicted_name}')
plt.show()

# Pattern: visually similar clothes are often confused.
# Improvement: use data augmentation to make the CNN generalize better.
