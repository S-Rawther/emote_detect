# Importing Numpy for Arrays and Pandas for Dataset Training
import numpy as np
import pandas as pa

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import np_utils

dataset = pa.read_csv('fer2013.csv')
trainX, trainY, testX, testY = [], [], [], []

for indexing, row in dataset.iterrows():
    value = row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
            trainX.append(np.array(value, 'float32'))
            trainY.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
            testX.append(np.array(value, 'float32'))
            testY.append(row['emotion'])
    except:
        print(f"Error Occurred while Indexing :{indexing} and row:{row}")

features = 64
num_labels = 7
b_size = 64
epochs = 40
width, height = 48, 48

trainX = np.array(trainX, 'float32')
trainY = np.array(trainY, 'float32')
testX = np.array(testX, 'float32')
testY = np.array(testY, 'float32')

trainY = np_utils.to_categorical(trainY, num_classes=num_labels)
testY = np_utils.to_categorical(testY, num_classes=num_labels)

trainX -= np.mean(trainX, axis=0)
trainX /= np.std(trainX, axis=0)

testX -= np.mean(testX, axis=0)
testX /= np.std(testX, axis=0)

trainX = trainX.reshape(trainX.shape[0], 48, 48, 1)

testX = testX.reshape(testX.shape[0], 48, 48, 1)

# Creating Convolutional Neural Network Layer
cnn_model = Sequential()

# 1st CNN Layer
cnn_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(trainX.shape[1:])))
cnn_model.add(Conv2D(64, kernel_size= (3, 3), activation='relu'))

cnn_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
cnn_model.add(Dropout(0.5))

# 2nd CNN Layer
cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(Conv2D(64, (3, 3), activation='relu'))

cnn_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
cnn_model.add(Dropout(0.5))

# 3rd CNN Layer
cnn_model.add(Conv2D(128, (3, 3), activation='relu'))
cnn_model.add(Conv2D(128, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

cnn_model.add(Flatten())

# Merging Neural Network
cnn_model.add(Dense(1024, activation='relu'))
cnn_model.add(Dropout(0.2))
cnn_model.add(Dense(1024, activation='relu'))
cnn_model.add(Dropout(0.2))

cnn_model.add(Dense(num_labels, activation='softmax'))

# Compiling CNN Model
cnn_model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

# Training the Model for Later Use
cnn_model.fit(trainX, trainY, batch_size=b_size, epochs=epochs, verbose=1, validation_data=(testX, testY), shuffle=True)

# Saving the model as .json file for detection
model_json = cnn_model.to_json()
with open("training.json", "w") as json_file:
    json_file.write(model_json)
cnn_model.save_weights("training.h5")

