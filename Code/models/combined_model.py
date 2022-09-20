# -*- coding: utf-8 -*-
"""
Created on Fri May 27 14:40:23 2022

@author: Student
"""

import torch
from torch import nn
import numpy as np
from odl.contrib import torch as odl_torch

class sandwich(nn.Module):
    def __init__(self,u_net_1,adj_op_layer,u_net_2):
        super(sandwich,self).__init__()
        self.u_net_1 = u_net_1
        self.adj_op_layer = adj_op_layer
        self.u_net_2 = u_net_2
    
    def forward(self,x):
        x = self.u_net_1(x).float()
        x = self.adj_op_layer(x.squeeze().unsqueeze(0)).float()
        x = self.u_net_2(x.unsqueeze(0)).float()
        return x

class Combined_model(nn.Module):
    def __init__(self,fw_model,fw_op,fw_swaped,adj_model,adj_op,adj_swaped):
        super(Combined_model,self).__init__()
        self.fw_model = fw_model
        self.fw_op = odl_torch.OperatorModule(fw_op)
        self.fw_swaped = fw_swaped
        self.adj_model = adj_model
        self.adj_op = odl_torch.OperatorModule(adj_op)
        self.adj_swaped = adj_swaped
    
    def forward(self,d):
        x = d[0].unsqueeze(0)
        y = d[1].unsqueeze(0)
        if self.fw_swaped:
            x = self.fw_model(x).float()
            x = self.fw_op(x).float()
        else:
            x = self.fw_op(x).float()
            x = self.fw_model(x).float()
        
        r = x - y
        if self.adj_swaped:
            r = self.adj_model(r).float()
            r = self.adj_op(r).float()
        else:
            r = self.adj_op(r).float()
            r = self.adj_model(r).float()
            
        return r
    
class Multiple_Iterations_model(nn.Module):
    def __init__(self,num_iter,step_function,fw_model,fw_op,fw_swaped,adj_model,
                 adj_op,adj_swaped):
        super(Multiple_Iterations_model,self).__init__()
        self.num_iter = num_iter
        self.step_function = step_function
        self.comb_model = Combined_model(fw_model, fw_op, fw_swaped, adj_model, adj_op, adj_swaped)
        
    def forward(self,d):
        x = d[0]
        y = d[1]
        d_i = [x,y]
        for i in range(self.num_iter):
            r = self.comb_model(d_i)
            self.step_function(x,y,r)
            d_i = [x,y]
        
        return d_i 
            
class Minus_y(nn.Module):
    def __init__(self,y):
        super(Minus_y,self).__init__()
        if type(y)  == np.ndarray:
            y = torch.from_numpy(y)
        self.y = y
        
    def forward(self,x):
        return x - self.y
    
    