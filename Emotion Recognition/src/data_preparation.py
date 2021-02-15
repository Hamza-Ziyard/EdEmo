import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

# prepare folder structure to store the png files of thr dataset
main_folders = ["test", "train"]
sub_folders = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
os.makedirs('data', exist_ok=True)
for main_folder in main_folders:
    os.makedirs(os.path.join('data', main_folder), exist_ok=True)
    for sub_folder in sub_folders:
        os.makedirs(os.path.join('data', main_folder, sub_folder), exist_ok=True)

# To store the count of each emotion category in both training set and testing set
angry = 0
disgusted = 0
fearful = 0
happy = 0
sad = 0
surprised = 0
neutral = 0
angry_test = 0
disgusted_test = 0
fearful_test = 0
happy_test = 0
sad_test = 0
surprised_test = 0
neutral_test = 0

# Reading the dataset from the csv file
dataset = pd.read_csv('./fer2013.csv')
dimensionalArray = np.zeros((48, 48), dtype=np.uint8)
print("Saving images...")

# read the csv file line by line
for i in tqdm(range(len(dataset))):
    test = dataset['pixels'][i]
    words = test.split()

    # the image size is 48x48
    for j in range(2304):
        xind = j // 48
        yind = j % 48
        dimensionalArray[xind][yind] = atoi(words[j])

    currentImage = Image.fromarray(dimensionalArray)

    # train
    if i < 28709:
        if dataset['emotion'][i] == 0:
            currentImage.save('train/angry/im' + str(angry) + '.png')
            angry += 1
        elif dataset['emotion'][i] == 1:
            currentImage.save('train/disgusted/im' + str(disgusted) + '.png')
            disgusted += 1
        elif dataset['emotion'][i] == 2:
            currentImage.save('train/fearful/im' + str(fearful) + '.png')
            fearful += 1
        elif dataset['emotion'][i] == 3:
            currentImage.save('train/happy/im' + str(happy) + '.png')
            happy += 1
        elif dataset['emotion'][i] == 4:
            currentImage.save('train/sad/im' + str(sad) + '.png')
            sad += 1
        elif dataset['emotion'][i] == 5:
            currentImage.save('train/surprised/im' + str(surprised) + '.png')
            surprised += 1
        elif dataset['emotion'][i] == 6:
            currentImage.save('train/neutral/im' + str(neutral) + '.png')
            neutral += 1

    # test
    else:
        if dataset['emotion'][i] == 0:
            currentImage.save('test/angry/im' + str(angry_test) + '.png')
            angry_test += 1
        elif dataset['emotion'][i] == 1:
            currentImage.save('test/disgusted/im' + str(disgusted_test) + '.png')
            disgusted_test += 1
        elif dataset['emotion'][i] == 2:
            currentImage.save('test/fearful/im' + str(fearful_test) + '.png')
            fearful_test += 1
        elif dataset['emotion'][i] == 3:
            currentImage.save('test/happy/im' + str(happy_test) + '.png')
            happy_test += 1
        elif dataset['emotion'][i] == 4:
            currentImage.save('test/sad/im' + str(sad_test) + '.png')
            sad_test += 1
        elif dataset['emotion'][i] == 5:
            currentImage.save('test/surprised/im' + str(surprised_test) + '.png')
            surprised_test += 1
        elif dataset['emotion'][i] == 6:
            currentImage.save('test/neutral/im' + str(neutral_test) + '.png')
            neutral_test += 1

print("Successfully converted the csv file to png files!")
