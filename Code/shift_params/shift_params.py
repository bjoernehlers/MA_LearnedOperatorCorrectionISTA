# -*- coding: utf-8 -*-
"""
Created on Wed May  4 15:19:51 2022

@author: Student
"""

import numpy as np
from util import check_path
 
shift_params = np.array([
    [[0.01,1000,0],[0.01,1000,0]],
    [[0.001,2000,0],[0.05,1000,2]],
    [[0.001,3000,7],[0.01,2000,0]],
    [[0.001,4000,0],[0.01,500,0]],
    [[0.001,5000,5],[0.001,2000,3]],
    [[0.001,6000,0],[0.0001,500,0]],
    [[0.001,7000,0],[0.01,5000,0]],
    ])
path = "shift_params/"
check_path(path)
np.save(file = path+"7_u_v_shift_2.npy", arr = shift_params)
