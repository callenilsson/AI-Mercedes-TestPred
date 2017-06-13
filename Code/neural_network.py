import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import pandas as pd

#Custom R2 loss used in Kaggle (Didn't work that well?)
def r2_score(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return -1.0 * (1 - SS_res / (SS_tot + K.epsilon()))

#Load Data
train = pd.read_csv('../Data/train.csv')
test = pd.read_csv('../Data/test.csv')
trainX = train.drop('y', 1).drop('ID', 1).as_matrix()
trainY = train.y.as_matrix()
testX = test.drop('ID', 1).as_matrix()

#Get unique strings from data as dictionary
chars = {}
for i in range(0, len(trainX)):
    for j in range(0, 8):
        chars[trainX[i,j]] = 0
for i in range(0, len(testX)):
    for j in range(0, 8):
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
#This makes the train/test len(chars_vec)=54 * 8 times wider though
trainX2 = np.zeros((len(trainX), len(trainX[0]) - 8 + 8*len(chars_vec) + 8))
testX2 = np.zeros((len(testX), len(testX[0]) - 8 + 8*len(chars_vec) + 8))
for i in range(0, len(trainX)):
    for j in range(0, 8):
        trainX2[i, j*len(chars_vec) : (j+1)*len(chars_vec)] = chars[trainX[i,j]]
        trainX2[i, 8*len(chars_vec) : len(trainX2[0]) - 8] = trainX[i,8:]
for i in range(0, len(testX)):
    for j in range(0, 8):
        testX2[i, j*len(chars_vec) : (j+1)*len(chars_vec)] = chars[testX[i,j]]
        testX2[i, 8*len(chars_vec) : len(testX2[0]) - 8] = testX[i,8:]

#EXTRA FEATURE
# Calculate mean time based on car model (Mean of Y based on X0 - X8)
charsYMeanVec = []
charsY = {}
charsCount = {}
for i in range(0,8):
    charsYMeanVec.append({})
    #Reset dictionaries
    for key in chars:
        charsY[key] = 0
        charsCount[key] = 0
        charsYMeanVec[i][key] = 0
    #Calculate sum of Y and its occurences
    for j in range(0, len(trainX)):
        charsCount[trainX[j,i]] += 1
        charsY[trainX[j,i]] += trainY[j]
    #Calculate mean of Y
    for key in charsCount:
        try:
            charsYMeanVec[i][key] = charsY[key] / charsCount[key]
        except:
            b = 0

#Add the Y means to final vector
for i in range(0,len(trainX2)):
    for j in range(0,8):
        trainX2[i, j + len(trainX2[0]) - 8] = charsYMeanVec[j][trainX[i,j]]
for i in range(0, len(trainX2)):
    for j in range(0, 8):
        testX2[i, j + len(testX2[0]) - 8] = charsYMeanVec[j][testX[i, j]]

#Hyperparameters
n_features = len(trainX2[0])
epochs = 15
batch_size = 100
dropout = 0.5
n_neurons = 4000
n_hidden_layers = 4

#Neural network
model = Sequential()
model.add(Dense(n_neurons, input_dim=n_features, activation='relu'))
model.add(Dropout(dropout))
for i in range(0,n_hidden_layers):
    model.add(Dense(n_neurons, activation='relu'))
    model.add(Dropout(dropout))
model.add(Dense(1, activation='relu'))

model.compile(loss='poisson', optimizer='adam', metrics=['accuracy'])

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
    f.write(str(test.ID[counter]) + "," + str(val) + "\n")
    counter += 1
f.close()
print("Done!")