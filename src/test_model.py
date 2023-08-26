from keras.models import load_model
import cv2
import numpy as np
import tensorflow as tf

model_path = "../model/note_classifier_model.h5"
sample_path = "../Data/samples"

model = load_model(model_path)

img = cv2.imread("2.png", 0)
img = tf.image.resize(np.expand_dims(img, axis=2), (256, 256))

pred = model.predict(np.expand_dims(img / 255, 0))


def get_note_type(pred_class: int):
    pred_class_map = {0: "Eight", 1: "Half", 2: "Quarter", 3: "Sixteenth", 4: "Whole"}
    if pred_class < 0 or pred_class > 4:
        raise Exception("Prediction index out of bounds")
    return pred_class_map[pred_class]


print(get_note_type(pred.argmax(axis=1)[0]))
