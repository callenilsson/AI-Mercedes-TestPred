import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import keras.backend as K
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import model_from_json

#Custom R2 loss used in Kaggle
def r2_score(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return SS_res / SS_tot

#Load Data
print("Loading data...")
train = pd.read_csv('../Data/train.csv')
test = pd.read_csv('../Data/test.csv')
trainX = train.drop('y', 1).drop('ID', 1).as_matrix()
trainY = train.y.as_matrix()
testX = test.drop('ID', 1).as_matrix()
multipliers = np.load('multipliers.npy')

#Remove outliers from training data
print("Removing outliers... ")
count = 0
prev_length = len(trainX)
for i in range(0,len(trainY)):
    try:
        if trainY[i] > 200:
            trainX = np.delete(trainX, (i), axis=0)
            trainY = np.delete(trainY, (i), axis=0)
            count += 1
            i -= 1
    except:
        b = 0
print(count, "of", prev_length, "outliers removed")

#Get unique strings from data as dictionary
print("One hot encoding data...")
n_string_features = 8
chars = {}
for i in range(0, len(trainX)):
    for j in range(0, n_string_features):
        chars[trainX[i,j]] = 0
for i in range(0, len(testX)):
    for j in range(0, n_string_features):
        chars[testX[i, j]] = 0

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
#This makes the train/test len(chars_vec)=54 * 8 = 432 dimensions wider though
trainX2 = np.zeros((len(trainX), len(trainX[0]) - n_string_features + n_string_features*len(chars_vec)))
testX2 = np.zeros((len(testX), len(testX[0]) - n_string_features + n_string_features*len(chars_vec)))
for i in range(0, len(trainX)):
    for j in range(0, n_string_features):
        trainX2[i, j*len(chars_vec) : (j+1)*len(chars_vec)] = chars[trainX[i,j]]
        trainX2[i, n_string_features*len(chars_vec) : len(trainX2[0])] = trainX[i,n_string_features:]
for i in range(0, len(testX)):
    for j in range(0, n_string_features):
        testX2[i, j*len(chars_vec) : (j+1)*len(chars_vec)] = chars[testX[i,j]]
        testX2[i, n_string_features*len(chars_vec) : len(testX2[0])] = testX[i,n_string_features:]

#Hyperparameters
n_features = len(trainX2[0])
epochs = 100
batch_size = 1403
dropout = 0.5
n_neurons = 2000
n_hidden_layers = 2
train = True

model = None
if train:
    #Neural network
    model = Sequential()
    model.add(Dense(n_neurons, input_dim=n_features, activation='relu'))
    model.add(Dropout(dropout))
    for i in range(0,n_hidden_layers):
        model.add(Dense(n_neurons, activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))

    model.compile(loss=r2_score, optimizer='adam')

    #Train
    model.fit(trainX2, trainY, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)

    #Save model
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved model to disk")

if not train:
    #Load pre-trained model
    print("Loading pre-trained model...")
    json_file = open("model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("model.h5")

#Plot Y and predicted Y
print('Plotting Y and predicted Y...')
trainYPred = model.predict(trainX2)
fig = plt.figure()
plt.scatter(trainYPred, trainY, s=5)
plt.xlabel('Y_pred')
plt.ylabel('Y')
plt.axis((70,180,70,180))
plt.show()

multipliers2 = np.ones((len(trainY)))
for i in range(0, len(trainY)):
    if abs(trainY[i] - trainYPred[i]) >= 2:
        multipliers2[i] = abs(int(trainY[i] - trainYPred[i]))
np.save('multipliers2', multipliers2)

#Test
print('Applying model to test set...')
predictions = model.predict(testX2)

#Write the test predictions to submission file
print("Writing test results to file...")
f = open("../Data/keras_predictions", 'w')
f.write("ID,y\n")
counter = 0
for val in np.nditer(predictions):
    f.write(str(test.ID[counter]) + "," + str(val) + "\n")
    counter += 1
f.close()
print("Done!")