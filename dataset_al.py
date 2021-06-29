from skimage.feature import local_binary_pattern
from skimage.io import imread
import numpy as np
import os
from al import *

def image_features(filename, b, P=7, R=16):
    print("Image Feature: {}".format(filename))

    # image = imread(filename, as_gray=True) # Imagem em escala de cinza
    image = imread(filename) # Imagem RGB
    image   = image[:,:,0] # Somente Vermelho
    
    lbp = local_binary_pattern(image, P, R)
    
    c = melhor_c(lbp, b)
    # b = c
    # c = melhor_c(lbp.T, b)
    
    # bins = 2 ** P
    # hist = np.histogram(lbp, bins=bins)[0]
    
    return c

# CRIANDO
# X = []
# idx = 150
# b = np.loadtxt("b.csv", delimiter=",")

# for i in range(idx, 190 + 1):
#     f = "Test/No_Fire/resized_test_nofire_frame{}.jpg".format(i)
#     if os.path.isfile(f):
#         c = image_features(f, b)
#         # print(c)
#         X.append(c)
# np.savetxt("Xtest_no.csv", X, fmt="%f", delimiter=",")

# CONCATENANDO
Xfire = np.loadtxt("Xtest_fire.csv", delimiter=",")
# Xfire = Xfire.astype(int)
Xno = np.loadtxt("Xtest_no.csv", delimiter=",")
# Xno = Xno.astype(int)

y = [ True for i in range(len(Xfire)) ] + [ False for i in range(len(Xno)) ]

X = np.concatenate([Xfire, Xno])
print(X.shape)
y = np.array(y)

np.savetxt("Xtest.csv", X, fmt="%f", delimiter=",")
np.savetxt("ytest.csv", y, fmt="%d", delimiter=",")
