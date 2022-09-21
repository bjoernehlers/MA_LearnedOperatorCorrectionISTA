    # """
    # creates a random sparcer matrix shift dataset depending on the parameters
    # """

import numpy as np
from scipy import sparse
from ray_transforms import get_ray_trafo
from util import rand_shift_params,sparse_Mat,check_path,get_shift

x_res = 64
y_res = 64
n_ang = 256
n_dtp = 96

rand_shift_dict = dict()
rand_shift_dict.update({"num_params":5})
rand_shift_dict.update({"amp_range":[0.03,0.05]})
rand_shift_dict.update({"freq_range":[500,5000]})
rand_shift_dict.update({"freq_shift_range":[0,2*np.pi]})
rand_shift_dict



path = 'Matritzen/64_64_256_96_5addet_shift_cpu/'
check_path(path)
for i in range(100):
    shift_params = rand_shift_params(**rand_shift_dict)
    shift = get_shift(shift_params)
    np.save(file = path + f'shift_params_{i}', arr = shift_params) 
    ray_trafo = get_ray_trafo(x_res,y_res,n_ang,n_dtp,DET_SHIFT = shift,detector_len=2,impl='astra_cpu')
    A_s = sparse_Mat(ray_trafo,x_res,y_res,n_ang,n_dtp)
    sparse.save_npz(path+f'ray_trafo_{i}.npz',A_s)