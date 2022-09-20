# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 21:11:13 2022

@author: Student
"""
import os
import matplotlib.pyplot as plt
import time
import odl
import numpy as np
from tqdm import tqdm
from scipy import sparse
import random

def hide_axis(fig):
    for x in fig.get_axes():
        x.get_xaxis().set_visible(False)
        x.get_yaxis().set_visible(False)
        
def check_path(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        
def save_loss_graph(train_loss,val_loss,title,path,doent_show = True):
    '''
    

    Parameters
    ----------
    train_loss : numpy array
        1_Dim array
    val_loss : numpy array
        1_Dim array
    title : str
        title displayd on the plot
    path : str
        save path including name and fileending (save\path\name.fileending)
    doent_show : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    '''
    fig, axs = plt.subplots()
    axs.plot(train_loss)
    axs.plot(val_loss)
    axs.set_title(title)
    fig.savefig(path)
    if doent_show:
        plt.close(fig)
    else:
        plt.show(fig)
        
def time_estimate(t:float, itr:int, t_num_itr:int):
    """time estimation for a loop where the n+1 loop takes aproxematly the time of the nth loop and the first loop together

    Args:
        t (float): time since the start of the first iteration (t_start -t_now)
        itr (int): the ieration number that just has finished the fisrt iteration is 1 not 0, or use the number of the next iteration step
        t_num_itr (int): the total number of iterations

    Returns:
        float: estimated remaining time
    """
    new_t_1_est = (2*t)/(itr**2+itr)
    t_remain = new_t_1_est * (t_num_itr**2+t_num_itr-itr**2-itr)/2
    return t_remain 

class added_estimate():
    def __init__(self):
        self.times = list()
    def est_func(self,t,itr,t_num_iter):
        if  len(self.times)>=2:
            dt = t + self.times[-2] - 2 * self.times[-1]
            t_iter = t - self.times[-1]
            iter_remain = t_num_iter - itr
            self.times.append(t)
            print(t_iter/60)
            return iter_remain * t_iter + dt*iter_remain*(iter_remain+1)/2
        elif len(self.times) == 0:
            self.times.append(t)
            return t*t_num_iter 
        elif len(self.times) == 1:
            self.times.append(t)
            return (t-self.times[0])*(t_num_iter-1)

class Timer():
    def __init__(self, num_iterations, est_func = time_estimate, path = None):
        self.t_0 = time.time()
        self.num_iterations = num_iterations
        self.est_func = est_func
        self.path = path 
        if path == None:
            self.write_to_file = False
        else:
            self.write_to_file = True
            check_path(path)


    def make_estimate(self,iteration, notes = ''):
        t = time.time()
        t_e = self.est_func(t-self.t_0,iteration,self.num_iterations)
        print(f'eta:{time.asctime(time.localtime(t+t_e))} in {t_e/3600:.1f}h')
        if self.write_to_file : 
            file = open(f'{self.path}eta.txt','a')
            file.write(f'iteration:{iteration:3} | ')
            file.write(f'eta: {time.asctime(time.localtime(t+t_e))} in {t_e/3600:.1f}h | ')
            file.write(f'{(t-self.t_0)/60} minutes since start'  )
            file.write(notes + '\n')
            file.close()

    def finished(self):
        t_end = time.time() 
        print(f'finisched at {time.asctime(time.localtime(t_end))} after {(t_end - self.t_0)/3600:.1f}h')
        if self.write_to_file:
            file = open(f'{self.path}eta.txt','a')
            file.write(f'finisched at {time.asctime(time.localtime(t_end))} after {(t_end-self.t_0)/3600:.1f}h')
            file.close()
        if self.write_to_file:
            os.rename(f'{self.path}eta.txt',f'{self.path}finisched.txt')

def plt_from_dic(a, title = None, save_path = None, doent_show = False):
    fig, ax = plt.subplots()
    ax.set_title(title)
    for e in a:
        ax.plot(a.get(e),label=e)
    ax.legend()
    if not save_path is None:
        fig.savefig(save_path)
    if doent_show:
        plt.close(fig)
    else:
        plt.show(fig)

def to_np_array(a):
    if type(a) == odl.discr.discr_space.DiscretizedSpaceElement:
            a = a.asarray()
    return a

def get_op(A_s:np.array,a:int,b:int,c:int,d:int):
    """returns operator taking axb np.arrays to cxd arrays:
    axd --> a*d --> A_s@a*d = c*d --> cxd
    Args:
        A_s (np.array): 2D-(Sparse)Matrix array
        a (int): input row size
        b (int): input collum size
        c (int): output row size
        d (int): output collum size

    Returns:
        op: Operator that has the Matrix and its adjoint/transpose impleamented + reshaping
    """
    def op(x):
        x = x.reshape(a*b)
        def adj_op(x):
            x  = x.reshape(c*d)
            return (A_s.T@x).reshape(a,b)
        op.adjoint = adj_op
        return (A_s@x).reshape(c,d)
    op(np.zeros((a,b)));
    return op

def Mat(lin_op,a:int,b:int,c:int,d:int):
    """
    takes the ray tranformation that is mapped from axb to cxd and makes the coseponding a*bxc*d radon matrix
    Args:
        lin_op (_type_): Linear Operator that schould be A Metrix be constructed from
        a (int): input row size
        b (int): input collum size
        c (int): output row size
        d (int): output collum size

    Returns:
        np.array: The a*b x c*d Matrix
    """
    X = np.eye(a*b)
    A = np.zeros((c*d,a*b))
    for i,x in enumerate(tqdm(X)):
        A_i = lin_op(x.reshape(a,b))
        if type(A_i) is not np.ndarray:
            A_i = A_i.asarray()
        A_i = A_i.reshape(c*d)
        A[:,i] = A_i
    return A

def sparse_Mat(lin_op,a:int,b:int,c:int,d:int,as_torch_tensor=False):
    """
    takes the ray tranformation that is mapped from axb to cxd and makes the coseponding a*bxc*d radon matrix
    Args:
        lin_op (_type_): Linear Operator that schould be A Metrix be constructed from
        a (int): input row size
        b (int): input collum size
        c (int): output row size
        d (int): output collum size
        as_torch_tensor (bool): if True gives back a pytorch sparse csr-tensor else a scipy csr-matrix. Default is True.

    Returns:
        scipy.sparse.csr_matrix:  or torch.sparse_csr_tensor depending on as_torch_tensor .a*b x c*d Matrix
    """
    # X = sparse.eye(a*b)
    A = sparse.lil_matrix((c*d,a*b))
    x = np.zeros(a*b)
    for i in tqdm(range(a*b)):
        x[i] = 1
        A_i = lin_op(x.reshape(a,b))
        x[i] = 0
        if type(A_i) is not np.ndarray:
            A_i = A_i.asarray()
        A[:,i] = A_i.reshape(c*d)
    A = sparse.csr_matrix(A)
    if as_torch_tensor:
        return torch.sparse_csr_tensor(A.indptr,A.indices,A.data)
    else:
        return A

def plots(h:int,w:int,r=1,sf = 3):
    """
    better plot layout
    Args:
        h (int): number of rows of plots
        w (int): number of collums of plots
        r (int, optional): the ratio of the plots width/height of olots. Defaults to 1.
        sf (int, optional): the scaling factor 1 = sf inches. Defaults to 3.

    Returns:
        _type_: the fig and axes from plt.subplots
    """
    fig,axs = plt.subplots(h,w)
    fig.set_size_inches(w*(sf*r),h*sf)
    # fig.tight_layout()
    return fig,axs

def rand_shift_params(num_params = 10,amp_range = [0.02,0.03],freq_range = [500,5000],freq_shift_range = [0,2*np.pi]):
    """generate random parametrs for creating sum of sinus shifts

    Args:
        num_params (int, optional): numner of added sinus shifts. Defaults to 10.
        amp_range (list, optional): possible amplitude intervall that is uniformly chossen from. Defaults to [0.02,0.03].
        freq_range (list, optional): possible frequenzy intervall that is uniformly chossen from. Defaults to [500,5000].
        freq_shift_range (list, optional): possible possible frequenzy shift intervall that is uniformly chossen from. Defaults to [0,2*np.pi].

    Returns:
        list: shift params [[u_shift_1,v_shift_1],...,[u_shift_N,v_shift_N]] with N = num_params and the u and v shifts are of the form [amplitude,frequenzy,frequenzy shift]
    """
    shift_params = list()
    for j in range(num_params):
        amplitude = random.uniform(*np.array(amp_range)/num_params)
        freq = random.uniform(*freq_range)
        freq_shift = random.uniform(*freq_shift_range)
        u_shift = [amplitude,freq,freq_shift]
        amplitude = random.uniform(*np.array(amp_range)/num_params)
        freq = random.uniform(*freq_range)
        freq_shift = random.uniform(*freq_shift_range)
        v_shift = [amplitude,freq,freq_shift]
        shift_params.append([u_shift,v_shift])
    return np.array(shift_params)

def error_for_y(y,e_p):
    """
    Returns an y with aded noise of size e

    Parameters
    ----------
    y : numpy array
        In put y.
    e_p : float
        controlls the size of the error term in percent. (y-y_e)/||y|| = e_p

    Returns
    -------
    y_e : numpy array 
         y with aded noise of size e_r.

    """
    e = np.random.rand(y.size)
    e = e/np.linalg.norm(e)
    e = e.reshape(y.shape)
    y_norm = np.linalg.norm(y.reshape(y.size)) 
    y_e = y + e_p*y_norm*e
    return y_e

def get_shift(shift_params: np.array):
    """Uses a list of parameters to create angle dependen u and v shifts 

    Args:
        shift_params (np.array): List of parameters [[[a_u_1,b_u_1,c_u_1],[a_v_1,b_v_1,c_v_1]],...,[[a_u_N,b_u_N,c_u_N],[a_v_N,b_v_N,c_v_N]]]
                                 with a amplitude b frequency and c freauquency shift
    Returns:
        shift funktion
    """
    def shift(angle):
        u_shift = 0
        v_shift = 0
        for i in range(shift_params.shape[0]):
            u_shift = u_shift + shift_params[i,0,0]*np.sin(shift_params[i,0,1]*angle+shift_params[i,0,2])
            v_shift = v_shift + shift_params[i,1,0]*np.sin(shift_params[i,1,1]*angle+shift_params[i,1,2])
        return np.array([u_shift,v_shift]).T
    return shift 

def im_norm(x:np.array):
    """ takes the L2 norm elements of the R^N that are arragend in more dimensions for examples images
    Args:
        x (np.array): image or block

    Returns:
        float : the 2 norm
    """
    x = x.reshape(x.size)
    return np.linalg.norm(x,2)

def sp(a,b):
    """takes to numpy arrays with the same number of elements and takes the L2 skalar produkt

    Args:
        a (np.array): _description_
        b (np.array): _description_

    Returns:
        float: _description_
    """
    return a.reshape(a.size)@b.reshape(b.size)