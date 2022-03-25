import scipy.io
import numpy as np
mat = scipy.io.loadmat ("C:\\Users\\junda\\OneDrive\\Bureau\\breastw.mat")
print(type(mat))

X = mat['X'] #On extrait la donnée du champ 'X' du dictionnaire
(np.shape(X)) #shape() de la bibliothèque numpy (ici np), renvoie la taille de la matrice (array numpy) en argument