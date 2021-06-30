import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pickle

Xtrain = np.loadtxt("Xtraining.csv", delimiter=",")
Xtrain = Xtrain.astype(int)
print(Xtrain)
print(Xtrain.shape)
ytrain = np.loadtxt("ytraining.csv", delimiter=",")
ytrain = ytrain.astype(bool)
print(ytrain)
print(ytrain.shape)

scaler = StandardScaler()
scaler.fit(Xtrain)

model = MLPClassifier(hidden_layer_sizes=(256, 128),
                      activation='relu',
                      solver='adam',
                      max_iter=1000,
                      random_state=17,
                      early_stopping=True,
                      validation_fraction=0.1)
model.fit(scaler.transform(Xtrain), ytrain)

pickle.dump(model, open("model.mlp", 'wb'))
