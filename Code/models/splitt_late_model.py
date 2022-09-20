# -*- coding: utf-8 -*-
"""
Created on Wed May 11 11:29:00 2022

@author: Student
"""

import torch
from torch import nn
from odl.contrib import torch as odl_torch
import numpy as np

def get_split_late_model(ray_trafos,kernel_size = 3,num_of_downs = 2,
                    start_channels_power = 3, in_channels = 1, out_channels = 1,
                    up_transpose = True):
    if (kernel_size % 2 == 1):
        return My_split_late_UNet(ray_trafos,in_channels, out_channels, kernel_size, 
                          start_channels_power, num_of_downs,
                          up_transpose)
    else:
        raise ValueError("kernel_size needs to be uneven")

class My_split_late_UNet(nn.Module):
    def __init__(self,ray_trafos,in_channels = 1, out_channels = 1
                 , kernel_size = 5,start_channels_power = 5, num_of_downs = 2,
                 up_transpose = True):
        super(My_split_late_UNet,self).__init__()
        self.num_of_downs = num_of_downs
        
        self.down = nn.ModuleList()
        self.down.append(ConvolutionBlock(in_channels, 2**start_channels_power,
                                          kernel_size))
        for i in range(0,num_of_downs):
            p = start_channels_power + i
            self.down.append(DownBlock(2**p, 2**(p+1)))
            
        self.up = nn.ModuleList()
        for i in range(num_of_downs,1,-1):
            p = start_channels_power + i
            self.up.append(UpBlock(2**p, 2**(p-1), kernel_size, up_transpose))
            
        self.ups = nn.ModuleList()
        for j in range(len(ray_trafos)):
                p = start_channels_power + 1
                self.ups.append(UpBlock(2**p, 2**(p-1), kernel_size, up_transpose))
        
        self.outs = nn.ModuleList()
        for j in range(len(ray_trafos)):
            self.out = nn.Conv2d(2**start_channels_power, out_channels,
                                 kernel_size = 1)
            self.outs.append(self.out)
        
        self.ray_trafo_layers = list()
        for rt in ray_trafos:
            self.ray_trafo_layers.append(odl_torch.OperatorModule(rt))
        
        
        # self.ups = list()
        # for j in range(len(ray_trafos)):
        #     self.up = nn.ModuleList()
        #     for i in range(num_of_downs,0,-1):
        #         p = start_channels_power + i
        #         self.up.append(UpBlock(2**p, 2**(p-1), kernel_size, up_transpose))
        #     self.ups.append(self.up)
        # self.outs = list()
        # for j in range(len(ray_trafos)):
        #     self.out = nn.Conv2d(2**start_channels_power, out_channels,
        #                          kernel_size = 1)
        #     self.outs.append(self.out)
        # self.ray_trafo_layers = list()
        # for rt in ray_trafos:
        #     self.ray_trafo_layers.append(odl_torch.OperatorModule(rt))
        

    # def forward(self,x):
    #     l = [] 
    #     for i in range(self.num_of_downs):
    #         l.append(self.down[i](x))
    #         x = l[-1]
    #     x = self.down[self.num_of_downs](x)
    #     y = list()
    #     for rt_layer in self.ray_trafo_layers:
    #         #x_j = x
    #         x_j = self.up[0](l[-(0+1)],x)
    #         for i in np.arange(self.num_of_downs-1)+1:
    #             x_j = self.up[i](l[-(i+1)],x_j)
    #         x_j = self.out(x_j)
    #         y.append(rt_layer(x_j))
    #     return torch.stack(y)
    
    def forward(self,x):
        l = [] 
        for i in range(self.num_of_downs):
            l.append(self.down[i](x))
            x = l[-1]
        x = self.down[self.num_of_downs](x)
        y = list()
        
        for i in range(self.num_of_downs-1):
               x = self.up[i](l[-(i+1)],x)
            
        j = 0
        for rt_layer in self.ray_trafo_layers:
            x_j = x
            x_j = self.ups[j](l[0],x)
            x_j = self.outs[j](x_j)
            y.append(rt_layer(x_j))
            j = j+1
        return torch.stack(y,dim = 3).squeeze(0)

    
class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 5):
        super(ConvolutionBlock,self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.convolution_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size, stride = 1, padding = pad),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size, stride = 1, padding = pad),
            nn.ReLU(),
            ) 
    def forward(self,x):
        x = self.convolution_block(x)
        return x 


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 5):
        super(DownBlock,self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.down_block = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size, stride = 1, padding = pad),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size, stride = 1, padding = pad),
            nn.ReLU()
            ) 
        
    def forward(self,x):
            return self.down_block(x)

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
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size, stride = 1, padding = pad),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size, stride = 1, padding = pad),
            nn.ReLU()
            )
    def forward(self,x_0,x_1):
        x_1 = self.up_conv(x_1)
        x = torch.cat((x_0,x_1),1)
        return self.conv(x)
    
# class SplitUp(nn.Module):
#     def __init__(self,num_rts,power, in_channels, out_channels, kernel_size = 5, 
#                  up_transpose = True):
#         super(SplitUp,self).__init__()
#         self.ups = nn.ModuleList()
#         for j in range(num_rts):
#             p = power
#             self.ups.append(UpBlock(2**p, 2**(p-1), kernel_size, up_transpose))
        