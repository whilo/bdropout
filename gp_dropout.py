import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import mnist

from maunaloa import load_mauna

srng = RandomStreams()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 1/(shape[1]+1)))

def rectify(X):
    return T.maximum(X, 0.)

def sigmoid(X):
    return 1/(1+T.exp(-X))

def dropout(X, p):
    mask = srng.binomial(n=1, p=1-p, size=X.shape)
    return X * T.cast(mask, theano.config.floatX)

# https://gist.github.com/Newmu/acb738767acb4788bac3
def Adam(cost, params, lr=0.001, b1=0.1, b2=0.001, e=1e-8):
    updates = []
    grads = T.grad(cost, params)
    i = theano.shared(floatX(0.))
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates



def model(X,
          w_h, w_h2, w_h3, w_h4, w_h5, w_o,
          b_h, b_h2, b_h3, b_h4, b_h5,
          p_drop=0.5, activation=rectify):
    X = dropout(X, p_drop)
    # rescale weights by dropout factor
    h = activation(T.dot(X, w_h) + b_h)
    h = dropout(h, p_drop)
    h2 = activation(T.dot(h, w_h2) + b_h2)
    h2 = dropout(h2, p_drop)
    h3 = activation(T.dot(h2, w_h3) + b_h3)
    h3 = dropout(h3, p_drop)
    h4 = activation(T.dot(h3, w_h4) + b_h4)
    h4 = dropout(h4, p_drop)
    h5 = activation(T.dot(h4, w_h5) + b_h5)
    h5 = dropout(h5, p_drop)
    py_x = T.dot(h5, w_o)
    return py_x

#trX, teX, trY, teY = load_mauna()
import h5py
f = h5py.File("data/train.h5")
trX = f["data"]
trY = f["label"]
f = h5py.File("data/test.h5")
teX = f["data"]
teY = f["label"]

rindex = range(len(trX))
trX = trX[rindex]
trY = trY[rindex]

X = T.fmatrix()
Y = T.fmatrix()

n_hidden = 1024

w_h = init_weights((1, n_hidden))
w_h2 = init_weights((n_hidden, n_hidden))
w_h3 = init_weights((n_hidden, n_hidden))
w_h4 = init_weights((n_hidden, n_hidden))
w_h5 = init_weights((n_hidden, n_hidden))
w_o = init_weights((n_hidden, 1))

b_h = theano.shared(floatX(np.zeros(n_hidden)))
b_h2 = theano.shared(floatX(np.zeros(n_hidden)))
b_h3 = theano.shared(floatX(np.zeros(n_hidden)))
b_h4 = theano.shared(floatX(np.zeros(n_hidden)))
b_h5 = theano.shared(floatX(np.zeros(n_hidden)))
#b_o = theano.shared(floatX(np.zeros(1)))


p_dropout = 0.1
noise_py_x = model(X,
                   w_h, w_h2, w_h3, w_h4, w_h5, w_o,
                   b_h, b_h2, b_h3, b_h4, b_h5,
                   p_drop=p_dropout,
                   activation=rectify)
py_x = model(X,
             w_h, w_h2, w_h3, w_h4, w_h5, w_o,
             b_h, b_h2, b_h3, b_h4, b_h5,
             p_drop=p_dropout, activation=rectify)

N = len(trX)
w_decay = 0.00001 # TODO same as in convnetjs code
l = 1.0
tau = l**2 * (1 - p_dropout) / (2 * N * w_decay)
cost = T.mean((noise_py_x - Y)**2) \
       + (1-p_dropout)*w_decay * (T.sum(w_h**2) + T.sum(w_h2**2) + T.sum(w_h3**2)
                                  + T.sum(w_h4**2) + T.sum(w_h5**2) + T.sum(w_o**2)
                                  + T.sum(b_h**2) + T.sum(b_h2**2) + T.sum(b_h3**2)
                                  + T.sum(b_h4**2) + T.sum(b_h5**2))
params = [w_h, w_h2, w_h3, w_h4, w_h5, w_o, b_h, b_h2, b_h3, b_h4, b_h5]
updates = Adam(cost, params)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=py_x, allow_input_downcast=True)


import matplotlib.pyplot as plt

def plot():
    l = len(trX)+len(teX)
    t = np.arange(l)
    preds = [predict(np.concatenate([trX,teX])) for i in range(100)]
    mu = np.mean(preds, axis=0).reshape(l)
    var = np.var(preds, axis=0).reshape(l) + tau**-1
    sigma = np.sqrt(var)

    fig, ax = plt.subplots(1)
    ax.plot(np.arange(len(trX)), trY, lw=2, label='data', color='red')
    ax.plot(t, mu, lw=2, label='fit', color='blue')
    ax.fill_between(t, mu+2*sigma, mu-2*sigma, facecolor='blue', alpha=0.5)
    ax.set_title('$\mu$ and $\pm 2\sigma$ interval')
    ax.legend(loc='upper left')
    ax.set_xlabel('month')
    ax.set_ylabel('concentration normalized')
    ax.grid()
    fig.show()




batch_size = len(trX) # speeds up, full batch training

weights_zero = []
weights_four = []
weights_five = []
for i in range(30000):
    for start, end in zip(range(0, len(trX)+1, batch_size), range(batch_size, len(trX)+1, batch_size)):
        cost = train(trX[start:end], trY[start:end])
    print i, ": ", np.mean((trY - predict(trX))**2), np.mean((teY - predict(teX))**2)
    if i % 200 == 0:
        weights_zero.append(w_h.get_value())
        weights_four.append(w_h4.get_value())
        weights_five.append(w_h5.get_value())
    if i % 1000 == 0:
        plot()

#plot()


weight_zarr = np.array([w.flatten()[0:100] for w in weights_zero])
weight_farr = np.array([w.flatten()[0:100] for w in weights_four])
