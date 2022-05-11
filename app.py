import tensorflow as tf
from tensorflow.keras import models
import cv2
import numpy as np

def img_loading(path):
    img_test = cv2.imread(path)
    img_array = (255 - img_test) / 255.0
    img_array = tf.expand_dims(img_array, axis=0)# Create a batch

    return img_array


def digit_prediction(img_array):
    model = models.load_model("models/20220404-003109-0.99.h5")

    prediction = model.predict(img_array)

    return prediction

def score(predictions):
    class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    score = tf.nn.softmax(predictions)

    # for score in scores[:]:
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score[-1])], 100 * np.max(score[-1]))
    )

if __name__ == "__main__":
    # Lecture des fichiers du sous-répertoire de test

    test_digits = [f"test/{i}.png" for i in range(0, 10)]
    predictions = []

    for test_digit in test_digits:
        img_array = img_loading(test_digit)
        predictions.append(digit_prediction(img_array)[0])
        score(predictions)