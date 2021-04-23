import argparse
import cv2
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
import os
import matplotlib.pyplot as mat_plt
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Get commands from user
ap = argparse.ArgumentParser()
ap.add_argument("--modelMode", help="train/display")
modelMode = ap.parse_args().modelMode


# plots accuracy and loss curves
def plot_edEmo_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the edEmomodel_history
    """
    figure, axis = mat_plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axis[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'])
    axis[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'])
    axis[0].set_title('Model Accuracy')
    axis[0].set_ylabel('Accuracy')
    axis[0].set_xlabel('Epoch')
    axis[0].set_xticks(np.arange(1, len(model_history.history['accuracy']) + 1),
                      len(model_history.history['accuracy']) / 10)
    axis[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axis[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axis[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axis[1].set_title('Model Loss')
    axis[1].set_ylabel('Loss')
    axis[1].set_xlabel('Epoch')
    axis[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axis[1].legend(['train', 'val'], loc='best')
    figure.savefig('plot.png')
    mat_plt.show()


# Defining data directories
train_dir = 'data/train'
test_dir = 'data/test'

num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 50

# Convert the each pixel to 8 bit value
train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

# Get dataset from the directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical')

test_generator = val_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
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
if modelMode == "train":
    edEmoModel.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
    edEmoModelInfo = edEmoModel.fit_generator(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=test_generator,
        validation_steps=num_val // batch_size)
    plot_edEmo_model_history(edEmoModelInfo)
    edEmoModel.save_weights('model.h5')


# Facial detection using haarcascade classifiers
# emotions will be displayed on your face from the webcam feed
elif modelMode == "display":
    edEmoModel.load_weights('model.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # loading haarcascade classifiers
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dictionary = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # Start webcam feed
    video_capture = cv2.VideoCapture(0)

    # obtain the start time
    start_time = time.time()

    # data structures
    time_data = np.array([0])
    prediction_data = np.array([0])

    while True:
        # Capture each frame at a time
        ret, frame = video_capture.read()
        if not ret:
            break

    #     # converting rgb to grayscale
    #     grayscaleImages = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     faces = face_cascade.detectMultiScale(
    #         grayscaleImages,
    #         scaleFactor=1.3,
    #         minNeighbors=5)

    #     # Draw rectangle
    #     for (x, y, w, h) in faces:
    #         cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
    #         roi_gray = grayscaleImages[y:y + h, x:x + w]
    #         cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
    #         prediction = edEmoModel.predict(cropped_img)
    #         maximumIndex = int(np.argmax(prediction))
    #         cv2.putText(frame, emotion_dictionary[maximumIndex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
    #                     (255, 255, 255), 2,
    #                     cv2.LINE_AA)

    #     # Display frame
    #     cv2.imshow('Video', frame)

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # # When everything is done, release the capture
    # video_capture.release()
    # cv2.destroyAllWindows()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            np.set_printoptions(formatter={'float_kind': '{:f}'.format})
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = edEmoModel.predict(cropped_img)
<<<<<<< HEAD
            print(prediction)
            maximumIndex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dictionary[maximumIndex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2,
                        cv2.LINE_AA)
=======
>>>>>>> dccae58752d28e2b6ffd2915a6d61b606a170562

            # Calculating the prediction percentage
            maxindex = int(np.argmax(prediction))
            data = cropped_img.astype('float32')
            data /= 255
            probabilities = edEmoModel.predict_proba(data, verbose=1)[0]
            max_string = str(probabilities[maxindex].item() * 100)
            first_three = ' '+max_string[0:3] + '%'

            # Run timer
            current_time = round(time.time()-start_time,2)
            time.sleep(0.25)
            time_data = np.append(time_data,current_time)

            if maxindex in [3,5]:
                prediction_data = np.append(prediction_data, probabilities[maxindex].item() * 100)
                print('enagaged')
            else:
                prediction_data = np.append(prediction_data,probabilities[maxindex].item() * -100)
                print('not enageged')
            # checkWhetherEngaged(prediction_data,probabilities,maxindex)


            cv2.putText(frame, emotion_dictionary[maxindex] + first_three, (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    # display graph
    mat_plt.plot(time_data,prediction_data)
    mat_plt.xlabel("Time")
    mat_plt.ylabel("Engagement")
    mat_plt.show()

# def display_graph(time_data,prediction_data):
#     mat_plt.plot(time_data,prediction_data)
#     mat_plt.xlabel("Time")
#     mat_plt.ylabel("Engagement")
#     mat_plt.show()

# def checkWhetherEngaged(prediction_data,probabilities,maxindex):
#     if maxindex in [3,5]:
#         prediction_data = np.append(prediction_data, probabilities[maxindex].item() * 100)
#         print('enagaged')
#     else:
#         prediction_data = np.append(prediction_data,probabilities[maxindex].item() * -100)
#         print('not enageged')    

# def calculatePredictionPercentage(maxindex,prediction,probabilities):
#     maxindex = int(np.argmax(prediction))
#             data = cropped_img.astype('float32')
#             data /= 255
#             probabilities = edEmoModel.predict_proba(data, verbose=1)[0]
#             max_string = str(probabilities[maxindex].item() * 100)
#             first_three = ' '+max_string[0:3] + '%'