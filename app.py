from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
# Keras
# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import cv2 as cv
# Define a flask app
app = Flask(__name__)
os.environ['KMP_DUPLICATE_LIB_OK']='True'
model = Sequential()
model.add(Conv2D(64, (3, 3) ,input_shape=(100,100,1), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(1024, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.load_weights("model4_weights.h5")

def load_preds(file_path):
    img_array = cv.imread(file_path, 0)
    new_array = cv.resize(img_array, (100, 100))
    new_array = np.float32(new_array)
    X = np.array(new_array).reshape(-1, 100, 100, 1)
    preds = model.predict(X)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="car"
    elif preds==1:
        preds="truck"
    elif preds==2:
        preds="bus"
    else:
        preds="motorcycle"
    return preds


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        preds = load_preds(file_path)
        return preds
    return None


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)



