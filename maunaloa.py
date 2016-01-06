import numpy as np

import matplotlib.pyplot as plt


def load_mauna():
    y = np.loadtxt("./maunaloa-co2/maunaloa_clean.txt", usecols=xrange(1,13), skiprows=1)[7:-1].flatten()
    x = np.array(xrange(len(y)))
    mx = np.mean(x)
    stdx = np.std(x)
    x = (x - mx)/stdx
    m, c = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y)[0]
    y_lin = m*x + c
    y -= y_lin
    my = np.mean(y)
    stdy = np.std(y)
    y = (y - my)/stdy

    x = x.reshape(len(x), 1)
    y = y.reshape(len(y), 1)
    return x[0:200], x[200:400], y[0:200], y[200:400]

#plt.plot(load_mauna()[2].flatten())
