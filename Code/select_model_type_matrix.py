# -*- coding: utf-8 -*-
"""
Created on Mon May 30 18:35:19 2022

@author: Student
"""
import torch.nn as nn
import torch
from scipy import sparse


# import torchvision

from ray_transforms import get_static_ray_trafo, get_static_ray_trafo_angle_list
from util import sparse_Mat
from models.my_unet import get_my_unet_model, ChangingUNet
from models.splitt_model import get_split_model
from models.splitt_late_model import get_split_late_model
from models.combined_model import sandwich
from models.LinearModel import LinearModel

class sparse_Mat_Layer(nn.Module):
    # currently not possible to load this into cuda A needs to be already on the device
    def __init__(self,A,out_shape:tuple):
        super(sparse_Mat_Layer,self).__init__()
        self.A = A
        self.out_shape = out_shape

    def forward(self,x):
        x = x.reshape(self.A.shape[1],1)
        return torch.sparse.mm(self.A,x).reshape(self.out_shape).float()


class Mat_Layer(nn.Module):
    # A needs to be already on the device
    def __init__(self,A,out_shape:tuple):
        super(Mat_Layer,self).__init__()
        self.A = A.float()
        self.out_shape = out_shape

    def forward(self,x):
        x = x.reshape(self.A.shape[1],1).float()
        return torch.mm(self.A,x).reshape(self.out_shape).float()

def select_fwd_model_type(c,static_A_s):
    
    if c.forward_model_type == 'plain_u_net':
        model = get_my_unet_model(kernel_size=c.kernel_size, num_of_downs=c.num_of_downs,
                                  start_channels_power=c.start_channels_power)

    elif c.last_layer == 'static_ray_trafo' or c.forward_model_type == 'static_ray_trafo':
        model = get_my_unet_model(kernel_size=c.kernel_size, num_of_downs=c.num_of_downs,
                                  start_channels_power=c.start_channels_power)
        ray_trafo_layer = Mat_Layer(static_A_s,(1,1,c.num_angles,c.detector_points))
        model = nn.Sequential(model,ray_trafo_layer)
    
    elif c.forward_model_type == 'just_the_u_net':
        model = ChangingUNet((c.x_res,c.y_res),(c.num_angles,c.detector_points),
                                kernel_size=c.kernel_size, num_of_downs=c.num_of_downs,
                                start_channels_power=c.start_channels_power)
    elif c.forward_model_type == 'sandwich':
        u_net_1 = get_my_unet_model(kernel_size=c.kernel_size, num_of_downs=c.num_of_downs,
                                  start_channels_power=c.start_channels_power)
        u_net_2 = get_my_unet_model(kernel_size=c.kernel_size, num_of_downs=c.num_of_downs,
                                  start_channels_power=c.start_channels_power)
        ray_trafo_layer = Mat_Layer(static_A_s,(1,1,c.num_angles,c.detector_points))
        model = nn.Sequential(u_net_1,ray_trafo_layer,u_net_2)

    elif c.forward_model_type == 'linear':
        model = LinearModel((c.x_res,c.y_res),(c.num_angles,c.detector_points))

    # elif c.last_layer == 'split_static_ray_trafo' or c.forward_model_type == 'split_static_ray_trafo':
    #     ray_trafo_list = get_static_ray_trafo_angle_list(c.x_res,c.y_res,c.num_angles,c.detector_points)
    #     model = get_split_model(ray_trafo_list)
    
    # elif c.last_layer == 'split_late_static_ray_trafo' or c.forward_model_type == 'split_late_static_ray_trafo':
    #     ray_trafo_list = get_static_ray_trafo_angle_list(c.x_res,c.y_res,c.num_angles,c.detector_points)
    #     model = get_split_late_model(ray_trafo_list)
    return model

def select_adj_model_type(c,static_AT_s):
    
    if c.adj_model_type == 'plain_u_net':
        adjoint_model = get_my_unet_model(kernel_size=c.adj_kernel_size,
                                          num_of_downs=c.adj_num_of_downs,
                                          start_channels_power=c.adj_start_channels_power)
    elif c.adj_model_type == 'll_adj_static_ray_trafo':
        adjoint_model = get_my_unet_model(kernel_size=c.adj_kernel_size,
                                          num_of_downs=c.adj_num_of_downs,
                                          start_channels_power=c.adj_start_channels_power)
        adj_ray_trafo_layer = Mat_Layer(static_AT_s,(1,1,c.x_res,c.y_res))
        adjoint_model = nn.Sequential(adjoint_model,adj_ray_trafo_layer)
    elif c.adj_model_type == 'just_the_u_net':
         adjoint_model = ChangingUNet((c.num_angles,c.detector_points),(c.x_res,c.y_res),
                                kernel_size=c.kernel_size, num_of_downs=c.num_of_downs,
                                start_channels_power=c.start_channels_power)
    elif c.adj_model_type == 'sandwich':
        u_net_1 = get_my_unet_model(kernel_size=c.adj_kernel_size,
                                          num_of_downs=c.adj_num_of_downs,
                                          start_channels_power=c.adj_start_channels_power)
        u_net_2 = get_my_unet_model(kernel_size=c.adj_kernel_size,
                                          num_of_downs=c.adj_num_of_downs,
                                          start_channels_power=c.adj_start_channels_power)
        adj_ray_trafo_layer = Mat_Layer(static_AT_s,(1,1,c.x_res,c.y_res))
        adjoint_model = nn.Sequential(u_net_1,adj_ray_trafo_layer,u_net_2)

    elif c.adj_model_type == 'linear':
        adjoint_model = LinearModel((c.num_angles,c.detector_points),(c.x_res,c.y_res))
    
    return adjoint_model