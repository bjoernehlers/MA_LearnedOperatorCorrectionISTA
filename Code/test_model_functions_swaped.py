# -*- coding: utf-8 -*-
"""
function sfor testing the learned corrections
"""

import torch
import numpy as np
import odl
from util import to_np_array


def get_net_corected_operator(op, model,device ='cpu',swaped = False, out_as_np_array = True):
    """gives back an oerator for reconstruction

    Args:
        op (_type_): operator we want to correct needs to have an adjoint
        model (_type_): the model to correct with
        device (str, optional): device on where to exicute the model. Defaults to 'cpu'.
        swaped (bool, optional): parameter if the operator is already in the model or not. Defaults to False.
        out_as_np_array (bool, optional): if false gives back a odl space element. Defaults to True.

    Returns:
        the crrected operator
    """
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
    

