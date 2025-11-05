import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd

# Load model
model = tf.keras.models.load_model("traffic_sign_model.h5")

# Load label names
labels = pd.read_csv("labels.csv")  # id,sign_name

# Prediction function
def predict_sign(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    sign_name = labels.loc[labels['id'] == class_index, 'sign_name'].values[0]
    confidence = np.max(predictions) * 100

    print(f"🚦 Predicted Sign: {sign_name} ({confidence:.2f}% confidence)")

# Example
predict_sign("sample_sign.jpeg")
