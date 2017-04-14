from random import random
from random import randint
import numpy as np

def augment_data(x, y, z, xlen, ylen):
    x_ad = []
    y_ad = []
    z_ad = []
    xlen_ad = []
    ylen_ad = []
    for i in xrange(len(x)):
    # print type(z[i])
    # print z[i].shape
    # print z[i]
        if np.argmax(z[i]) == 1:
            outcome = random() <= 0.2
            if outcome:
                # augment
                id = randint(1, 3)
                if id == 1:
                    # use question 1
                    x_ad.append(x[i])
                    y_ad.append(x[i])
                    z_ad.append(z[i])
                    xlen_ad.append(xlen[i])
                    ylen_ad.append(ylen[i])
                elif id == 2:
                    # use question 2
                    x_ad.append(y[i])
                    y_ad.append(y[i])
                    z_ad.append(z[i])
                    xlen_ad.append(xlen[i])
                    ylen_ad.append(ylen[i])
                else:
                    # swap questions
                    x_ad.append(y[i])
                    y_ad.append(x[i])
                    z_ad.append(z[i])
                    xlen_ad.append(xlen[i])
                    ylen_ad.append(ylen[i])
            else:
                x_ad.append(x[i])
                y_ad.append(y[i])
                z_ad.append(z[i])
                xlen_ad.append(xlen[i])
                ylen_ad.append(ylen[i])
        else:
            x_ad.append(x[i])
            y_ad.append(y[i])
            z_ad.append(z[i])
            xlen_ad.append(xlen[i])
            ylen_ad.append(ylen[i])

    assert(len(x) == len(x_ad)), 'len(x) != len(x_ad)'
    assert(len(y) == len(y_ad)), 'len(y) != len(y_ad)'
    assert(len(z) == len(z_ad)), 'len(z) != len(z_ad)'

    return x_ad, y_ad, z_ad, xlen_ad, ylen_ad
