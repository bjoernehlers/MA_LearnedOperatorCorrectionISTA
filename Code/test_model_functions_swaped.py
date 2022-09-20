# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 16:48:18 2022

@author: Student
"""

import torch
import numpy as np
import odl
from util import to_np_array


def get_net_corected_operator(op, model,device ='cpu',swaped = False, out_as_np_array = True):
    get_cor_op = net_corected_operator(op, model,device,swaped, out_as_np_array)
    return get_cor_op.get_impl


class net_corected_operator:
    """
    class of operators that computes  r = model(op(x))
    """
    def __init__(self, op, model, device ='cpu',swaped = False, out_as_np_array = True):
        self.op = op
        self.model = model
        self.out_as_np_array = out_as_np_array
        self.device = device
        self.swaped = swaped
    
    def get_impl(self, x):
        self.model.eval()
        with torch.no_grad():
            if not self.swaped:
                y = self.op(x)
                y = to_np_array(y)
            else:
                y = x
            r = self.model(torch.from_numpy(y).unsqueeze(
                0).unsqueeze(0).to(self.device).float())
            if self.out_as_np_array:
                r = r.cpu().numpy()
        return r.squeeze()
    
class comb_net_corected_operator:
    """
    class of operators that computes  r = model(op(x))
    """
    def __init__(self, op, model, device ='cpu', out_as_np_array = True):
        self.op = op
        self.model = model
        self.out_as_np_array = out_as_np_array
        self.device = device
    
    def get_impl(self, x,y):
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(self.device).float()
            y = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).to(self.device).float()
            r = self.model([x,y])
            if self.out_as_np_array:
                r = r.cpu().numpy()
        return r.squeeze()

def get_alignment(true_op,true_adj_op,cor_op,cor_adj_op,y):
    return Alignment(true_op, true_adj_op, cor_op, cor_adj_op, y).get_impl
    
class Alignment:
    def __init__(self,true_op,true_adj_op,cor_op,cor_adj_op,y,
                 norm = np.linalg.norm):
        self.true_op = true_op
        self.true_adj_op = true_adj_op
        self.cor_op = cor_op
        self.cor_adj_op = cor_adj_op
        self.y = y
        self.norm = norm
    
    def get_impl(self,x):
        a = self.true_adj_op(self.true_op(x)-self.y).asarray()
        a = a.reshape(a.size)
        b = self.cor_adj_op(self.cor_op(x)-self.y)
        if type(b) == odl.discr.discr_space.DiscretizedSpaceElement:
            b = b.asarray()
        b = b.reshape(b.size)
        cos_Phi = (a@b)/(self.norm(a)*self.norm(b))
        return cos_Phi


# class Grad_Desc2():
#     def __init__(self,mu,lam,grad_R,op,y,device,fw_model,fw_swaped,adj_model,adj_swaped):
#         self.mu = mu
#         self.lam = lam
#         self.grad_R = grad_R
#         self.y = y
#         self.fw_op = get_net_corected_operator(op, fw_model, device, fw_swaped)
#         self.adj_op = get_net_corected_operator(op.adjoint, adj_model, device, 
#                                                 adj_swaped)
        
#     def get_impl(self,x):
#         r = self.fw_op(x) - self.y
#         F_abl = self.adj_op(r)
#         x = x - self.mu*(F_abl + self.lam * self.grad_R(x).asarray())
#         np.maximum(x,0)
#         return x

        
class Grad_Desc():
    def __init__(self,mu,lam,grad_R,op,y,device,fw_model,fw_swaped,adj_model,adj_swaped):
        self.mu = mu
        self.lam = lam
        self.grad_R = grad_R
        self.y = y
        self.fw_op = get_net_corected_operator(op, fw_model, device, fw_swaped)
        self.adj_op = get_net_corected_operator(op.adjoint, adj_model, device, 
                                                adj_swaped)
        
    def get_impl(self,x):
        r = self.fw_op(x) - self.y
        F_abl = self.adj_op(r)
        x = x - self.mu*(F_abl + self.lam * self.grad_R(x).asarray())
        x = np.maximum(x,0)
        return x

class ISTA():
    def __init__(self,mu,lam,op,y,device,fw_model,fw_swaped,adj_model,adj_swaped):
        self.mu = mu
        self.lam = lam
        self.y = y
        self.fw_op = get_net_corected_operator(op, fw_model, device, fw_swaped)
        self.adj_op = get_net_corected_operator(op.adjoint, adj_model, device, 
                                                adj_swaped)
    
    def soft_shrink(self,x,alpha):
        return np.sign(x) * np.maximum(np.abs(x)-alpha,0)

    def get_impl(self,x):
        r = self.fw_op(x) - self.y
        F_abl = self.adj_op(r)
        x = self.soft_shrink(x-self.mu/self.lam * F_abl, self.mu)
        # x = np.maximum(x,0)
        return x

        
# class Test():
#     def __init__(self,phantom,e_p,step_op,true_op,fw_model,adj_model):

        
        