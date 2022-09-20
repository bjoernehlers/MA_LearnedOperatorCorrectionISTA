# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 12:51:23 2022

@author: Student

updated on 20.6.2022
partially folllowing :https://medium.com/analytics-vidhya/unet-implementation-in-pytorch-idiot-developer-da40d955f201
"""

import torch
from torch import nn
import numpy as np

def get_my_unet_model(kernel_size = 3,num_of_downs = 2,start_channels_power = 3,
                      in_channels = 1, out_channels = 1,
                      up_transpose = True):
    if (kernel_size % 2 == 1):
        return MyUNet(in_channels, out_channels, kernel_size, 
                          start_channels_power, num_of_downs,
                          up_transpose)
    else:
        raise ValueError("kernel_size needs to be uneven")

class MyUNet(nn.Module):
    def __init__(self,in_channels = 1, out_channels = 1, kernel_size = 5,
                 start_channels_power = 5, num_of_downs = 2,
                 up_transpose = True):
        super(MyUNet,self).__init__()
        self.num_of_downs = num_of_downs
        
        self.down = nn.ModuleList()
        self.down.append(ConvolutionBlock(in_channels, 2**start_channels_power,
                                          kernel_size))
        for i in range(0,num_of_downs):
            p = start_channels_power + i
            self.down.append(DownBlock(2**p, 2**(p+1)))
                 
        self.up = nn.ModuleList()
        for i in range(num_of_downs,0,-1):
            p = start_channels_power + i
            self.up.append(UpBlock(2**p, 2**(p-1), kernel_size, up_transpose))
            
        self.out = nn.Conv2d(2**start_channels_power, out_channels,
                             kernel_size = 1)
        
    def forward(self,x):
        l = [] 
        for i in range(self.num_of_downs):
            l.append(self.down[i](x))
            x = l[-1]
        x = self.down[self.num_of_downs](x)
        for i in range(self.num_of_downs):
            x = self.up[i](l[-(i+1)],x)
        x = self.out(x)
        return x

    
class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 5):
        super(ConvolutionBlock,self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.convolution_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size, stride = 1, padding = pad),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size, stride = 1, padding = pad),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            ) 
    def forward(self,x):
        x = self.convolution_block(x)
        return x 


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 5):
        super(DownBlock,self).__init__()
        self.pool = nn.MaxPool2d((2,2))
        self.conv = ConvolutionBlock(in_channels, out_channels, kernel_size)
        
    def forward(self,x):
        x = self.pool(x)
        x = self.conv(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 5, 
                 up_transpose = True):
        super(UpBlock,self).__init__()
        pad = int((kernel_size - 1) / 2)
        # self.up_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size = 1),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #     )
        if up_transpose :
            self.up_conv = nn.ConvTranspose2d(in_channels, out_channels,
                                             kernel_size = 2, stride= 2)
        else:
            self.up_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
        
        self.conv = ConvolutionBlock(in_channels, out_channels, kernel_size)
    def forward(self,x_0,x_1):
        x_1 = self.up_conv(x_1)
        x = torch.cat((x_0,x_1),1)
        return self.conv(x)


class ChangeBlock(nn.Module):
    def __init__(self,in_shape = (64,64), out_shape = (256,96),
                 start_channels_power = 5, current_num_of_downs = 2):
        super(ChangeBlock,self).__init__()
        s = 2**(current_num_of_downs)
        # if in_shape[0]%s != 0 or in_shape[1]%s != 0 or out_shape[0]%s != 0 or  out_shape[1]%s != 0:
        #     raise ValueError("input and output dimensions need to be divisiable by 2^num_of_downs")
        layer_in_shape = []
        for x in in_shape:
            layer_in_shape.append(x//s)
        layer_out_shape = []
        for x in out_shape:
            layer_out_shape.append(x//s)
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

class ChangingUNet(nn.Module):
    def __init__(self,in_shape,out_shape,in_channels = 1, out_channels = 1, kernel_size = 5,
                 start_channels_power = 5, num_of_downs = 2,
                 up_transpose = True):
        super(ChangingUNet,self).__init__()
        # s = 2**(num_of_downs)
        # if in_shape[0]%s != 0 or in_shape[1]%s != 0 or out_shape[0]%s != 0 or  out_shape[1]%s != 0:
        #     raise ValueError("input and output dimensions need to be divisiable by 2^num_of_downs")
        self.num_of_downs = num_of_downs
        self.down = nn.ModuleList()
        self.change = nn.ModuleList()
        self.down.append(ConvolutionBlock(in_channels, 2**start_channels_power,
                                          kernel_size))
        for i in range(0,num_of_downs):
            p = start_channels_power + i
            self.down.append(DownBlock(2**p, 2**(p+1)))
            self.change.append(ChangeBlock(in_shape,out_shape,start_channels_power,i))
        self.change.append(ChangeBlock(in_shape,out_shape,start_channels_power,num_of_downs))

        self.up = nn.ModuleList()
        for i in range(num_of_downs,0,-1):
            p = start_channels_power + i
            self.up.append(UpBlock(2**p, 2**(p-1), kernel_size, up_transpose))
            
        self.out = nn.Conv2d(2**start_channels_power, out_channels,
                             kernel_size = 1)
        
    def forward(self,x):
        l = [] 
        for i in range(self.num_of_downs):
            x = self.down[i](x)
            l.append(self.change[i](x))
            
        x = self.down[self.num_of_downs](x)
        x = self.change[self.num_of_downs](x)

        for i in range(self.num_of_downs):
            x = self.up[i](l[-(i+1)],x)
        x = self.out(x)
        return x