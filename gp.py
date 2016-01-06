# adapted from http://nbviewer.ipython.org/github/SheffieldML/notebook/blob/master/GPy/basic_gp.ipynb

import numpy as np
import GPy
from maunaloa import load_mauna


#trX, teX, trY, teY = load_mauna()
import h5py
f = h5py.File("data/train.h5")
trX = f["data"]
trY = f["label"]
f = h5py.File("data/test.h5")
teX = f["data"]
teY = f["label"]


kernel = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=0.01)

m = GPy.models.GPRegression(trX,trY,kernel)

fig = m.plot()
GPy.plotting.show(fig)
