import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pickle

Xtest = np.loadtxt("Xtest.csv", delimiter=",")
Xtest = Xtest.astype(int)
print(Xtest)
print(Xtest.shape)
ytest = np.loadtxt("ytest.csv", delimiter=",")
ytest = ytest.astype(bool)
print(ytest)
print(ytest.shape)

model = pickle.load(open("model.mlp", 'rb'))

Xtrain = np.loadtxt("Xtraining.csv", delimiter=",")
Xtrain = Xtrain.astype(int)
scaler = StandardScaler()
scaler.fit(Xtrain)
ypredicted = model.predict(scaler.transform(Xtest))
print("Acuracia:", accuracy_score(ypredicted, ytest))

ytrain = np.loadtxt("ytraining.csv", delimiter=",")
ytrain = ytrain.astype(bool)
print("Cross Score:", cross_val_score(model, Xtrain, ytrain))
