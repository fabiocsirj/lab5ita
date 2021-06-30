from skimage.feature import local_binary_pattern
from skimage.io import imread
import os
import numpy as np

rng = np.random.default_rng(seed=17)

def image_features(filename, P=7, R=16):
    print("Image Feature: {}".format(filename))

    # image = imread(filename, as_gray=True) # Imagem em escala de cinza
    image = imread(filename) # Imagem RGB
    image = image[:,:,0] # Somente Vermelho
    # image = image[:,:,1] # Somente Verde
    # image = red + green  # Captura amarelo
    
    lbp = local_binary_pattern(image, P, R)
    bins = 2 ** P
    hist = np.histogram(lbp, bins=bins)[0]
    
    return hist

def get_images(folder, maximages=None, rng=None):
    files = os.listdir(folder)
    if maximages is not None:
        assert rng is not None
        files = rng.choice(files, maximages)
        
    return [ image_features(os.path.join(folder, file)) for file in files ]

def get(dir):
    fire_dir    = "{}/Fire/".format(dir)
    no_fire_dir = "{}/No_Fire/".format(dir)
    Xfire = get_images(fire_dir)
    Xno   = get_images(no_fire_dir)

    y = [ True for i in range(len(Xfire)) ] + [ False for i in range(len(Xno)) ]

    X = np.concatenate([Xfire, Xno])
    y = np.array(y)

    return (X, y)

def save(X, y, file):
    X_file_save = "X{}.csv".format(file)
    y_file_save = "y{}.csv".format(file)
    np.savetxt(X_file_save, X, fmt="%d", delimiter=",")
    np.savetxt(y_file_save, y, fmt="%d", delimiter=",")


(X, y) = get("Training")
save(X, y, "training")

(X, y) = get("Test")
save(X, y, "test")
