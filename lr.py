"""
Return LR1 and LR2, resutlts quit close from the paper (except maybe the sem)

pytorch get nan if running with same archi than CEVAE.(tensroflow)
"""

from argparse import ArgumentParser

from initialisation import init_qz
from datasets import IHDP
from evaluation import Evaluator, get_y0_y1
from networks import p_x_z, p_t_z, p_y_zt, q_t_x, q_y_xt, q_z_tyx

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
from torch.distributions import normal
from torch import optim

dataset = IHDP()

def sqrt_pehe(mu1, mu0, ypred1, ypred0):
    return np.sqrt(np.mean(np.square((mu1 - mu0) - (ypred1 - ypred0))))

def get_train_data(train, valid, test, contfeats, binfeats):
    # read out data
    (xtr, ttr, ytr), (y_cftr, mu0tr, mu1tr) = train
    (xva, tva, yva), (y_cfva, mu0va, mu1va) = valid
    (xte, tte, yte), (y_cfte, mu0te, mu1te) = test

    # reorder features with binary first and continuous after
    perm = binfeats + contfeats
    xtr, xva, xte = xtr[:, perm], xva[:, perm], xte[:, perm]
    # concatenate train and valid for training
    xalltr, talltr, yalltr = np.concatenate([xtr, xva], axis=0), np.concatenate([ttr, tva], axis=0), np.concatenate(
        [ytr, yva], axis=0)

    return xalltr, talltr, yalltr, xte, tte


def lr1(xtr, ttr, ytr, xte, tte):
    # args: train & test
    from sklearn.linear_model import LinearRegression
  
    xttr = np.concatenate((xtr, ttr), axis = 1)
    lr = LinearRegression()
    lr.fit(xttr, ytr)


    xt0tr = np.concatenate((xtr,np.zeros(ttr.shape)), axis = 1)
    xt1tr = np.concatenate((xtr,np.zeros(ttr.shape) + 1), axis = 1)
    xt0_te = np.concatenate((xte,np.zeros(tte.shape)), axis = 1)
    xt1_te = np.concatenate((xte,np.zeros(tte.shape) + 1), axis = 1)    
    y0_tr = lr.predict(xt0tr)
    y1_tr = lr.predict(xt1tr)
    y0_te = lr.predict(xt0_te)
    y1_te = lr.predict(xt1_te)
    
    return y0_tr, y1_tr, y0_te, y1_te

def lr2(xtr, ttr, ytr, xte, tte):
    # args: train & test
    from sklearn.linear_model import LinearRegression
  
    xttr = np.concatenate((xtr, ttr), axis = 1)
    lr0 = LinearRegression()
    lr1 = LinearRegression()

    idx1, idx0 = np.where(ttr == 1)[0], np.where(ttr == 0)[0]
    x0tr = xtr[idx0]
    x1tr = xtr[idx1]
    y0tr = ytr[idx0]
    y1tr = ytr[idx1]
    lr0.fit(x0tr, y0tr)
    lr1.fit(x1tr, y1tr)

    y0_tr = lr0.predict(xtr)
    y1_tr = lr1.predict(xtr)
    y0_te = lr0.predict(xte)
    y1_te = lr1.predict(xte)
    
    return y0_tr, y1_tr, y0_te, y1_te