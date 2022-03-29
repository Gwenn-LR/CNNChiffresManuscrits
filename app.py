import tensorflow as tf
from tensorflow.keras import models
import cv2
import numpy as np

model = models.load_model("models/20220330-004526-0.99")

img_array = cv2.imread("test/0.png")

img_array = (255 - img_array) / 255.0

img_array = tf.expand_dims(img_array, axis=0) # Create a batch

predictions = model.predict(img_array)

class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
