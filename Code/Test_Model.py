import numpy as np

import torch

import matplotlib.pyplot as plt

from ray_transforms import get_ray_trafo, get_static_ray_trafo
from test_model_functions_swaped import get_net_corected_operator
# import random
import torch 

import os
from conf import config
from select_model_type_matrix import select_fwd_model_type,select_adj_model_type
from scipy import sparse
from util import get_op,Mat,plots,rand_shift_params,error_for_y

def im_norm(x):
    x = x.reshape(x.size)
    return np.linalg.norm(x,2)

def sp(a,b):
    return a.reshape(a.size)@b.reshape(b.size)

def Test(step_op,L,true_op,y_e,p,x_0,fw_op,adj_op,lam,mu,num_iter):
    F_abl = lambda x: adj_op(fw_op(x)-y_e)
    F_true_abl = lambda x: true_op.adjoint(true_op(x)-y_e)
    x = x_0
    loss = np.zeros(num_iter+1)
    loss[0] = im_norm(p-x)
    TM = np.zeros(num_iter)
    LL = np.zeros(num_iter)
    AL = np.zeros(num_iter)
    FwL = np.zeros(num_iter)
    AdL = np.zeros(num_iter)
    for i in range(num_iter):
        x_old = x
        x = step_op(x,F_abl(x))
        loss[i+1] = im_norm(p-x)
        T_mu = (x_old-x)/(mu)
        TM[i] = im_norm(T_mu)
        LL[i] = L(x_old)-L(x)
        AL[i] = mu*(sp(F_true_abl(x_old)-F_abl(x_old),T_mu)/lam+0.5*TM[i]**2)
        FwL[i] = im_norm(true_op(x)-fw_op(x))
        r = fw_op(x)-y_e
        AdL[i] = im_norm(true_op.adjoint(r)-adj_op(r))
    return {'loss':loss,'TM':TM,'LL':LL,'AL':AL,'FwL':FwL,'AdL':AdL}

def save_plot_Test(path,dic,background = {},show = False):
    fig,axs = plots(2,1,3/2)
    axs[0].set_title('reconstruction loss')
    axs[0].plot(background.get('static',[]),label = 'static')
    axs[0].plot(background.get('true',[]),label = 'true')
    axs[0].plot(dic.get('loss',[]),label = 'cor')
    axs[0].set_yscale('log')
    axs[0].legend()
    axs[1].set_title('alignement')
    axs[1].plot(dic.get('LL')/dic.get('TM')**2,label='LL')
    axs[1].plot(dic.get('AL')/dic.get('TM')**2,label='AL')
    axs[1].set_yscale('log')
    axs[1].legend()
    fig.savefig(path)
    if show:
        plt.show(fig)
    else:
        plt.close(fig)

