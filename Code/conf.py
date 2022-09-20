# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:50:44 2022

@author: Student
"""
import numpy as np
from scipy import sparse
from Iteration_Steps import GD_step,ISTA_step
import json


class config:
    """
    creates a class with the values extracted from a conf_dict, if a value is
    not in the conf_dict the valu will be set to None.
    """
    def __init__(self,run_name):
        self.run_name = run_name
        self.path = 'runs/'+self.run_name+'/'
        self.f =open(self.path + 'conf_dict.json','r')
        self.d = json.load(self.f)

        self.gpu_idx = self.d.get('gpu_index')
        #Paths
        self.run_name = self.d.get('run_name')
        self.note = self.d.get('note')
        self.path = self.d.get('path')
        self.fileending = self.d.get('fileending')
        self.phantoms_path = self.d.get('phantoms_path')
        self.train_phantom_path = self.d.get('train_phantom_path')
        self.val_phantom_path = self.d.get('val_phantom_path')
        self.test_phantom_path = self.d.get('test_phantom_path')
        if self.d.get('val_op_paths',False):
            self.val_op_list = list()
            for path in self.d.get('val_op_paths'):
                self.val_op_list.append(sparse.load_npz(path))
        
        #Display
        self.doent_show = self.d.get('doent_show')
        self.model_path = self.d.get('model_path')
        
        
        self.device = self.d.get('device')
        
        #Trainig
        
        self.start_iteration = self.d.get('start_iteration',0)
        self.num_iterations = self.d.get('num_iterations')
        self.num_epoch_list = self.d.get('num_epoch_list')
        self.num_adj_epoch_list = self.d.get('num_adj_epoch_list')
        self.batch_size = self.d.get('batch_size')
        
        #Defs
        
                
        self.step_type = self.d.get("step_type",'Ista')
        # if step_type == "GD":
        #     self.step_op = GD_step(self.mu,self.lam,self.grad_R)
        # elif step_type == "ISTA":
        #     self.step_op = ISTA_step(self.mu,self.lam)

        self.mu = self.d.get('mu')
        self.lam = self.d.get('lam')
        self.e_p = self.d.get('e_p')
        psd = self.d.get('pos_lam_dict')
        if psd['space_type']=='log':
            self.possible_lam = np.logspace(psd['min'],psd['max'])
        self.pos_constraint = self.d.get('pos_constraint')
        if self.pos_constraint is None:
            self.pos_constraint = False     
        self.R = self.d.get('R')
        self.grad_R = lambda x: x

        self.x_0_selection_rule = self.d.get('x_0_selection_rule','adj')
        self.train_set_selection_rule = self.d.get('train_set_selection_rule','last_and_random')#options are at the moment 'All','last_and_random'
        
        #Ray Trafos
        self.x_res = self.d.get('x_res')
        self.y_res = self.d.get('y_res')
        self.detector_points = self.d.get('detector_points')
        self.num_angles = self.d.get('num_angles')
        self.detector_len = self.d.get('detector_len')

        self.A_s_path = self.d.get('A_s_path')
        self.load_A_s = (self.A_s_path is not None)
        self.true_op_load_num = self.d.get('true_op_load_num',0)

        self.static_A_s_path = self.d.get('static_A_s_path')
        self.load_static_A_s = (self.static_A_s_path is not None)

        self.test_A_s_path = self.d.get('test_A_s_path')
        self.test_the_model = self.d.get('test_the_model',self.test_A_s_path is not None)
        
        # amplitude,frequifenzi,phaseschift
        self.just_one = self.d.get('just_one')
        if self.d.get('shift_params_path') is not None:
            self.shift_params = np.load(self.d.get('shift_params_path'))
        if self.d.get('shift_params_list_path') is not None:
            self.shift_params_list = np.load(self.d.get('shift_params_list_path'))
        else:
            self.shift_params_list=[]
        self.num_fixed = self.d.get('num_fixed')
        self.num_rand = self.d.get('num_rand')
        self.num_rand_phantoms = self.d.get('num_rand_phantoms')
        self.rand_phantoms_e_p = self.d.get('rand_phantoms_e_p')
        self.num_rand_ops_load = self.d.get('num_rand_ops_load',0)#before I only had new
        self.num_rand_ops_new = self.d.get('num_rand_ops_new',0)
        self.num_params_shift = self.d.get('num_params_shift',100)
        self.amp_range = self.d.get('amp_range', [0.03,0.05])
        self.freq_range = self.d.get('freq_range',[500,5000])
        self.freq_shift_range = self.d.get('freq_shift_range',[0,2*np.pi])
        self.use_true_op_xis = self.d.get('use_true_op_xis',False)
        
        
        
        self.fw_data_selection_rule = self.d.get('fw_data_selection_rule','All')
        self.adj_data_selection_rule = self.d.get('adj_data_selection_rule','All')
        self.max_len_dataset = self.d.get('max_len_dataset',5000)
        self.num_train_zeros = self.d.get('num_train_zeros',0)#number of [0,0] elements that are added to the forward and adjoint dataset this is in addition to the before createted datase, i.e. the dataset is the larger then max_len_dataset       
        
        #Model
        self.kernel_size = self.d.get('kernel_size')
        self.num_of_downs = self.d.get('num_of_downs')
        self.start_channels_power = self.d.get('start_channels_power')
        self.last_layer = self.d.get('last_layer') # 'static_ray_trafo'
        self.forward_model_type = self.d.get('forward_model_type')
        self.forward_swaped = self.d.get('forward_swaped')
        self.learning_rate = self.d.get('learning_rate')
        
        self.adj_kernel_size = self.d.get('adj_kernel_size')
        self.adj_num_of_downs = self.d.get('adj_num_of_downs')
        self.adj_model_type = self.d.get('adj_model_type')
        self.adj_swaped = self.d.get('adj_swaped')
        self.adj_start_channels_power = self.d.get('adj_start_channels_power')

        self.overwrite_model_saves = self.d.get('overwrite_model_saves',False)
        #loss_fn
        #optimizer
        #sheduler
        self.schedular_patientce = self.d.get('schedular_patientce',50000)
        self.adj_schedular_patientce = self.d.get('adj_schedular_patientce',50000)
        self.model_state_dict_path = self.d.get('model_state_dict_path')
        self.adjoint_model_state_dict_path = self.d.get('adjoint_model_state_dict_path')
        self.loops_per_iteation = self.d.get('loops_per_iteration',1)




