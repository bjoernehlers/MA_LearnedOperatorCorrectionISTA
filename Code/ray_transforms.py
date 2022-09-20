# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 12:15:54 2022
Playing around to get a Raytransform operater "true" and "static" both with adjoint
@author: Student
"""

import numpy as np
import odl


def get_static_ray_trafo(xresolution, yresolution,
                         NUM_ANGLES = 64, NUM_PARTION = 64,detector_len = 1.7,impl='astra_cpu'):
    # xresolution = np.size(phantoms,0)
    # yresolution = np.size(phantoms,1)
    
    xpoints = np.arange(xresolution)
    ypoints = np.arange(yresolution)
    

    grid_scaled = odl.discr.grid.RectGrid((xpoints+0.5)/xresolution-0.5,(ypoints+0.5)/yresolution-0.5)
    box_scaled = odl.set.domain.IntervalProd([-0.5,-0.5], [0.5,0.5])

    partition_scaled = odl.discr.partition.RectPartition(box_scaled,grid_scaled)
    space_scaled = odl.uniform_discr_frompartition(partition_scaled)# maybe chge the dtype to float32 (dtype = 'float32') 

    static_geometry = get_static_fan_geometry(NUM_ANGLES,NUM_PARTION,detector_len)
    return odl.tomo.RayTransform(space_scaled, static_geometry,impl=impl)

def get_static_fan_geometry(NUM_ANGLES, NUM_PARTION,detector_len = 1.7):
    
    #angel partition
    apart = odl.uniform_partition(0,2*np.pi,NUM_ANGLES)
    #detector partion
    dpart = odl.uniform_partition(-detector_len,detector_len,NUM_PARTION)
    #source radius
    src_radius = 5#np.sqrt(2)

    det_radius = 5#np.sqrt(2)

    return odl.tomo.geometry.FanBeamGeometry(apart, dpart, src_radius, det_radius)
    
def get_static_ray_trafo_angle_list(xresolution, yresolution,
                                    NUM_ANGLES = 64, NUM_PARTION = 64,detector_len=1.7):
    # xresolution = np.size(phantoms,0)
    # yresolution = np.size(phantoms,1)
    
    xpoints = np.arange(xresolution)
    ypoints = np.arange(yresolution)
    

    grid_scaled = odl.discr.grid.RectGrid((xpoints+0.5)/xresolution-0.5,(ypoints+0.5)/yresolution-0.5)
    box_scaled = odl.set.domain.IntervalProd([-0.5,-0.5], [0.5,0.5])

    partition_scaled = odl.discr.partition.RectPartition(box_scaled,grid_scaled)
    space_scaled = odl.uniform_discr_frompartition(partition_scaled)# maybe chge the dtype to float32 (dtype = 'float32') 

    static_geometry = get_static_fan_geometry(NUM_ANGLES,NUM_PARTION,detector_len)
    
    rt_l = list()
    for j in range(NUM_ANGLES):
        rt_l.append(odl.tomo.RayTransform(space_scaled, static_geometry[j]))
        
    return rt_l

# def get_ray_trafo(xresolution, yresolution,
#                   NUM_ANGLES = 128, NUM_PARTION = 128):
#     # xresolution = np.size(phantoms,0)
#     # yresolution = np.size(phantoms,1)

#     xpoints = np.arange(xresolution)
#     ypoints = np.arange(yresolution)


#     grid_scaled = odl.discr.grid.RectGrid(
#         (xpoints+0.5)/xresolution-0.5,(ypoints+0.5)/yresolution-0.5
#         )
#     box_scaled = odl.set.domain.IntervalProd([-0.5,-0.5], [0.5,0.5])

#     partition_scaled = odl.discr.partition.RectPartition(box_scaled,grid_scaled)
#     space_scaled = odl.uniform_discr_frompartition(partition_scaled)# maybe chge the dtype to float32 (dtype = 'float32') 

#     #angel partition
#     apart = odl.uniform_partition(0,2*np.pi,NUM_ANGLES)
#     #detector partion
#     dpart = odl.uniform_partition(-1.7,1.7,NUM_PARTION)
#     #source radius
#     src_radius = np.sqrt(2)

#     det_radius = np.sqrt(2)

#     det_shift=lambda angle: np.reshape(
#         np.array([angle*0, 0.05*np.sin(1000*angle)]).T, (NUM_ANGLES, 2)
#         )
#     geometry = odl.tomo.geometry.FanBeamGeometry(
#         apart, dpart, src_radius, det_radius, det_shift_func=det_shift
#         )
#     return odl.tomo.RayTransform(space_scaled, geometry)

def get_ray_trafo(xresolution, yresolution,
                  NUM_ANGLES = 64, NUM_PARTION = 64,
                  impl = 'astra_cpu',DET_SHIFT=lambda angle: 
                      np.array([angle*0, 0*angle]).T,detector_len=1.7):
    # xresolution = np.size(phantoms,0)
    # yresolution = np.size(phantoms,1)

    xpoints = np.arange(xresolution)
    ypoints = np.arange(yresolution)


    grid_scaled = odl.discr.grid.RectGrid(
        (xpoints+0.5)/xresolution-0.5,(ypoints+0.5)/yresolution-0.5
        )
    box_scaled = odl.set.domain.IntervalProd([-0.5,-0.5], [0.5,0.5])

    partition_scaled = odl.discr.partition.RectPartition(box_scaled,grid_scaled)
    space_scaled = odl.uniform_discr_frompartition(partition_scaled)# maybe chge the dtype to float32 (dtype = 'float32') 

    #angel partition
    apart = odl.uniform_partition(0,2*np.pi,NUM_ANGLES)
    #detector partion
    dpart = odl.uniform_partition(-detector_len,detector_len,NUM_PARTION)
    #source radius
    src_radius = 5#np.sqrt(2)

    det_radius = 5#np.sqrt(2)

    det_shift= lambda angle: DET_SHIFT(angle)
    geometry = odl.tomo.geometry.FanBeamGeometry(
        apart, dpart, src_radius, det_radius,
        det_shift_func=det_shift)
    #src_shift = det_shift
    # geometry = odl.tomo.geometry.FanBeamGeometry(
    #     apart, dpart, src_radius, det_radius,
    #     det_shift_func=det_shift, src_shift_func=src_shift)
    
    return odl.tomo.RayTransform(space_scaled, geometry,impl=impl)

   

class Ray_Trafo_c_Gamma:
    def __init__(self, ray_trafo, Gamma):
        self.ray_trafo = ray_trafo
        self.Gamma = Gamma
        
    def get_RT_c_Gamma(self, x):
        return self.ray_trafo(self.Gamma(x))
    
    
def get_RTcG(RT,G):
    def RTcG(x):
        return RT(G(x))
    # return lambda x: (RT(G(x)))
    return RTcG