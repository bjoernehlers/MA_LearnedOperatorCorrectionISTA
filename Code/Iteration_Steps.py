import numpy as np
import torch

def get_F_abl(A,y,adj_A = None):
    """gives the differential of 1/2 ||Ax-y||^2, i.e. A*(Ax-y) 

    Args:
        A (odl.operator): The Forward operator
        y (np.array): y
        adj_A (odl.operator, optional): adjoint operator if None A.adjoint is used.
         Defaults to None.

    Returns:
        function: A*(Ax-y) 
    """
    if adj_A is None:
        adj_A = A.adjoint
    def F_abl(x):
        x = adj_A(A(x)-y)
        return x.asarray()
    return F_abl

def get_F_abl_normed(A,y,adj_A = None,norm = None):
    """gives the differential of 1/2 ||Ax-y||^2, i.e. A*(Ax-y) 

    Args:
        A (odl.operator): The Forward operator
        y (np.array): y
        adj_A (odl.operator, optional): adjoint operator if None A.adjoint is used.
         Defaults to None.

    Returns:
        function: A*(Ax-y) 
    """
    if adj_A is None:
        adj_A = A.adjoint
    if norm is None:
        norm = A.norm(estimate=True)
    def F_abl(x):
        x = adj_A(A(x)-y)
        return norm**-2*x.asarray()
    return F_abl

def GD_step(mu,lam,grad_R,pos_constraint = False):
    if pos_constraint:
        def step(x,F_abl_x):
            x = x - mu*(F_abl_x-lam*grad_R(x).asarray())
            x = np.maximum(x,0)
            return x
    else:
        def step(x,F_abl_x):
            x = x - mu*(F_abl_x-lam*grad_R(x).asarray())
            return(x)
    return step 


def get_grad_desc_step(F_abl,mu,lam,grad_R,pos_constraint = True):
    """gives the gradient step

    Args:
        F_abl (function): The gradient of F
        mu (_type_): step length
        lam (_type_): cooefifziant of the 
        grad_R (odl.operator): _description_
        pos_constraint (bool, optional): enables a positivity constarint.
         Defaults to True.

    Returns:
        function: gradient iteration step
    """
    if pos_constraint:
        def step(x):
            x = x - mu*(F_abl(x)-lam*grad_R(x).asarray())
            x = np.maximum(x,0)
            return x
    else:
        def step(x):
            x = x - mu*(F_abl(x)-lam*grad_R(x).asarray())
            return(x)
    return step 


def soft_shrink(x,alpha):
    """the soft shrinkige operator
    Args:
        x (np.array): input
        alpha ( float ): parameter

    Returns:
        np.array: soft_schrink(x)
    """
    return np.sign(x) * np.maximum(np.abs(x)-alpha,0)

def ISTA_step(mu,lam):
    def step(x,F_abl_x):
        x = soft_shrink(x-mu/lam*F_abl_x,mu)
        return x
    return step

def get_ISTA_step(F_abl,lam,alpha):
    """ISTA iteration step

    Args:
        F_abl (function): gradient of F
        lam (_type_): stepsize
        alpha (_type_): softschrink parameter
    Returns:
        np.array: ISTA iteration step
    """
    def step(x):
        x = soft_shrink(x-lam*F_abl(x),lam*alpha)
        return x
    return step

