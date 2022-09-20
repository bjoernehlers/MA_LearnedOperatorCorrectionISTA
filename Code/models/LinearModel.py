import torch
from torch import nn
import numpy as np

class LinearModel(nn.Module):
    def __init__(self,in_shape = (64,64), out_shape = (256,96)):
        super(LinearModel,self).__init__()
        layer_in_shape = []
        for x in in_shape:
            layer_in_shape.append(x)
        layer_out_shape = []
        for x in out_shape:
            layer_out_shape.append(x)
        self.Flatten = nn.Flatten(-len(layer_in_shape))
        in_features = np.product(layer_in_shape)
        out_features = np.product(layer_out_shape)
        self.Linear = nn.Linear(in_features,out_features)
        self.Unflatten = nn.Unflatten(-1,layer_out_shape)
    def forward(self,x):
        x = self.Flatten(x)
        x = self.Linear(x)
        x = self.Unflatten(x)
        return x