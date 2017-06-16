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

# Split data in integer dictionary
print("Splitting data in integer dictionary...")
intDictY = {}
for i in range(0, 300):
    for j in range(0, len(trainY)):
        if i < trainY[j] and trainY[j] < i+1:
            try:
                intDictY[i].append(j)
            except:
                intDictY[i] = [j]
    try:
        temp = intDictY[i]
        intDictY[i] = np.zeros((len(trainY)))
        for j in temp:
            intDictY[i][j] = 1
    except:
        b = 0

#Balance the nbr of true/false samples for each integer
print("Balancing the nbr of true/false samples for each integer...")
intDictX = {}
for i in sorted(list(intDictY.keys())):
    tempXVec = []
    for j in range(0, len(intDictY[i])):
        if intDictY[i][j] == 1:
            tempXVec.append(trainX2[j])

    tempX = np.zeros((int(len(trainX2)/3), len(trainX2[0])))
    tempY = np.ones((int(len(trainX2)/3)))

    for j in range(0, len(tempX)):
        tempX[j,:] = tempXVec[j % len(tempXVec)]

    intDictX[i] = np.append(trainX2, tempX, axis=0)
    intDictY[i] = np.append(intDictY[i], tempY, axis=0)

    count = 0
    for j in intDictY[i]:
        if j == 1:
            count += 1

#First hyperparameters
n_features = len(trainX2[0])
epochs = 30
batch_size = 1403
dropout = 0.5
n_neurons = 800
n_hidden_layers = 2
model = None

#Neural network for each integer
firstResults = np.zeros((len(trainX2), len(intDictY.keys())))
firstModels = []
count = 0
for i in sorted(list(intDictY.keys())):
    print("Training model", count+1, "/", len(intDictY.keys()), "...", i)
    model = Sequential()
    model.add(Dense(n_neurons, input_dim=n_features, activation='relu'))
    model.add(Dropout(dropout))
    for j in range(0, n_hidden_layers):
        model.add(Dense(n_neurons, activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(intDictX[i], intDictY[i], epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
    firstModels.append(model)

    results = model.predict(trainX2)
    for j in range(0, len(results)):
        firstResults[j][count] = results[j][0]
    count += 1

#Final hyperparameters
n_features = len(trainX2[0])
epochs = 50
batch_size = 1403
dropout = 0.5
n_neurons = 1000
n_hidden_layers = 2
model = None

#Final neural network
print("Training final model...")
model = Sequential()
model.add(Dense(n_neurons, input_dim=len(firstResults[0]), activation='relu'))
model.add(Dropout(dropout))
for i in range(0,n_hidden_layers):
    model.add(Dense(n_neurons, activation='relu'))
    model.add(Dropout(dropout))
model.add(Dense(1, activation='linear'))

model.compile(loss=r2_score, optimizer='adam')

#Train
model.fit(firstResults, trainY, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)

f = open("../Data/test", 'w')
results = model.predict(firstResults)
for i in range(0, len(results)):
    f.write(str(trainY[i]) + ", " + str(results[i])+ " - ")
    for j in range(0, len(firstResults[0])):
        f.write(str(format(firstResults[i][j], '.1f')) + ", ")
    f.write("\n")
f.close()

#Plot Y and predicted Y
print('Plotting Y and predicted Y...')
trainYPred = model.predict(firstResults)
fig = plt.figure()
plt.scatter(trainYPred, trainY, s=5)
plt.xlabel('Y_pred')
plt.ylabel('Y')
plt.axis((70,180,70,180))
plt.show()

#Test
firstResults = np.zeros((len(testX2), len(intDictY.keys())))
for i in range(0, len(firstModels)):
    print("Applying model", i, "/", len(firstModels), "to test set...")
    results = firstModels[i].predict(testX2)
    for j in range(0, len(results)):
        firstResults[j,i] = results[j][0]
print('Applying final model to test set...')
predictions = model.predict(firstResults)

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