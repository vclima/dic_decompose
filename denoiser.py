import numpy as np

# proxl1
def proxl1(x,lamb):
    return np.sign(x)*np.maximum((np.abs(x)-lamb),np.zeros(x.shape))
# def proxl1_alt(x,lamb):
#    y = np.maximum(x - lamb, 0)
#    y = y + np.minimum(x + lamb, 0)
# return y

# proxl2 lamb/2 || . ||_2^2
def proxl2sq(x,lamb):
    return x/(1+lamb)
