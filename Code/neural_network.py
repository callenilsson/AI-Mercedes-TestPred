import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import pandas as pd

#Load Data
train = pd.read_csv('../Data/train.csv')
trainX = train.drop('y', 1).drop('ID', 1).as_matrix()
trainY = train.y.as_matrix()
testX = pd.read_csv('../Data/test.csv').drop('ID', 1).as_matrix()

#Get strings from data as dictionary
chars = {}
for i in range(0, len(trainX)):
    for j in range(0, 8):
        chars[trainX[i,j]] = 0

#One hot encode each string in the dictionary as numpy arrays
chars_vec = sorted(list(chars.keys()))
for key in chars:
    for i in range(0, len(chars_vec)):
        if key == chars_vec[i]:
            array = np.zeros(len(chars_vec))
            array[i] = 1
            chars[key] = array
            break

#Replace the 1-8 columns that has strings with one hot encoding vectors.
#This makes the train/test len(chars_vec)=54 * 8 times wider though
trainX2 = np.zeros((len(trainX), len(trainX[0]) - 8 + 8*len(chars_vec)))
testX2 = np.zeros((len(testX), len(testX[0]) - 8 + 8*len(chars_vec)))
for i in range(0, len(trainX)):
    for j in range(0, 8):
        trainX2[i, j*len(chars_vec) : (j+1)*len(chars_vec)] = chars[trainX[i,j]]
        trainX2[i, 8*len(chars_vec) : len(trainX2)] = trainX[i,8:]
for i in range(0, len(testX)):
    for j in range(0, 8):
        try:
            testX2[i, j*len(chars_vec) : (j+1)*len(chars_vec)] = chars[testX[i,j]]
            testX2[i, 8*len(chars_vec) : len(testX2)] = testX[i,8:]
        except:
            print("Missing one hot vector for:", testX[i,j])

#Hyperparameters
n_features = len(trainX2[0])
epochs = 1
batch_size = 1000
dropout = 0.5
n_neurons = 2000

#Neural network
model = Sequential()
model.add(Dense(n_neurons, input_dim=n_features, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(int(n_neurons/2), input_dim=n_features, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(1, activation='relu'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

#Train
model.fit(trainX2, trainY, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)

#Test
print('Applying model to test set...')
predictions = model.predict(testX2)

#Write the test predictions to submission file
print("Writing test results to file...")
f = open("../Data/keras_predictions", 'w')
f.write("ID,y\n")
counter = 0
for val in np.nditer(predictions):
    f.write(str(train.ID[counter]) + "," + str(val) + "\n")
    counter += 1
f.close()
print("Done!")