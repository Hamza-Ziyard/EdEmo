import argparse

import cv2
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow_core.python.keras.optimizers import Adam
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Get commands from user
ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="train/display")
mode = ap.parse_args().mode

# Defining data directories
train_dir = 'data/train'
test_dir = 'data/test'

num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 50

# Convert the each pixel to 8 bit value
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Get dataset from the directories
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

test_generator = val_datagen.flow_from_directory(
        test_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

# Creating the edEmoModel
edEmoModel = Sequential()

# Defining the convolutional layers of the cnn model
edEmoModel.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
edEmoModel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
edEmoModel.add(MaxPooling2D(pool_size=(2, 2)))
edEmoModel.add(Dropout(0.25))

edEmoModel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
edEmoModel.add(MaxPooling2D(pool_size=(2, 2)))
edEmoModel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
edEmoModel.add(MaxPooling2D(pool_size=(2, 2)))
edEmoModel.add(Dropout(0.25))

edEmoModel.add(Flatten())
edEmoModel.add(Dense(1024, activation='relu'))
edEmoModel.add(Dropout(0.5))
edEmoModel.add(Dense(7, activation='softmax'))

# If you want to train the same model or try other models, go for this



# Facial detection using haarcascade classifiers

# loading haarcascade classifiers
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dictionary = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Start webcam feed
video_capture = cv2.VideoCapture(0)

while True:
    # Capture each frame at a time
    ret, frame = video_capture.read()
    if not ret:
        break

    # converting rgb to grayscale
    grayscaleImages = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        grayscaleImages,
        scaleFactor=1.3,
        minNeighbors=5)

    # Draw rectangle
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray = grayscaleImages[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = edEmoModel.predict(cropped_img)
        maximumIndex = int(np.argmax(prediction))
        cv2.putText(frame, emotion_dictionary[maximumIndex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    # Display frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()


