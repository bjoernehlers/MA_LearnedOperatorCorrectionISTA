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
        self.run_name = run_name # the name zhis training run is reffered to
        self.path = 'runs/'+self.run_name+'/' 
        self.f =open(self.path + 'conf_dict.json','r')
        self.d = json.load(self.f)

        self.gpu_idx = self.d.get('gpu_index') # alows to set the gpu for Pytorch and astra
        #Paths
        self.run_name = self.d.get('run_name')
        self.note = self.d.get('note') 
        self.path = self.d.get('path')
        self.fileending = self.d.get('fileending','.png')  # fileending of the plots that saved
        self.phantoms_path = self.d.get('phantoms_path') # the path to saved numpy array of phantoms  fixed phatoms (currently unused)
        self.train_phantom_path = self.d.get('train_phantom_path') # the path to saved numpy array of train phantoms  fixed phatoms (currently unused)
        self.val_phantom_path = self.d.get('val_phantom_path') # the path to saved numpy array of validation phatoms
        self.test_phantom_path = self.d.get('test_phantom_path') # the path to saved numpy array of fixed phatoms 
        # if self.d.get('val_op_paths',False): # the path to saved sparse matricies  (currently unused)
        #     self.val_op_list = list()
        #     for path in self.d.get('val_op_paths'):
        #         self.val_op_list.append(sparse.load_npz(path))
        
        #Display
        self.doent_show = self.d.get('doent_show') # plots are not printed  (currently unused)
        self.model_path = self.d.get('model_path') # path there the models schould be saved
        
        
        self.device = self.d.get('device') # device that should be used
        
        #Trainig
        
        self.start_iteration = self.d.get('start_iteration',0) 
        self.num_iterations = self.d.get('num_iterations')
        self.num_epoch_list = self.d.get('num_epoch_list') # a list the num_epoch_list[i] is the number of epochs for training the forward model in iteration i  
        self.num_adj_epoch_list = self.d.get('num_adj_epoch_list') # the same as before only for adjoint
        self.batch_size = self.d.get('batch_size',1)
        #Defs
        
                
        self.step_type = self.d.get("step_type",'ISTA')# the type of algorithm that is used for reconstruction 'ISTA' or 'GD' The testing while training des not work on 'GD' 
        # if step_type == "GD":
        #     self.step_op = GD_step(self.mu,self.lam,self.grad_R)
        # elif step_type == "ISTA":
        #     self.step_op = ISTA_step(self.mu,self.lam)

        self.mu = self.d.get('mu') # step length mu
        self.lam = self.d.get('lam') #regularization parameter lambda
        self.e_p = self.d.get('e_p')
        psd = self.d.get('pos_lam_dict') # information where to search for an optial lam (currently unused)
        if psd['space_type']=='log':
            self.possible_lam = np.logspace(psd['min'],psd['max'])
        self.pos_constraint = self.d.get('pos_constraint') # toggles pos constraint (currently unused)
        if self.pos_constraint is None:
            self.pos_constraint = False     
        self.R = self.d.get('R') # name of or the regulizer (currently unused)
        self.grad_R = lambda x: x #(currently unused)

        self.x_0_selection_rule = self.d.get('x_0_selection_rule','adj')# selectin of x_0 'adj' \tilde{A}^*y_e or '0' 0_X
        self.train_set_selection_rule = self.d.get('train_set_selection_rule','last_and_random')#options are at the moment 'All','last_and_random'
        
        #Ray Trafos
        self.x_res = self.d.get('x_res') # x resolution of phantomas
        self.y_res = self.d.get('y_res') # y resolution of phantoms
        self.detector_points = self.d.get('detector_points') # number of detector points
        self.num_angles = self.d.get('num_angles') # number of angles
        self.detector_len = self.d.get('detector_len',2) #length of the detetor can be adjusted deppending on the shifts amplitude 

        self.A_s_path = self.d.get('A_s_path') # path to the sparse matrices with the shift
        self.load_A_s = (self.A_s_path is not None)
        self.true_op_load_num = self.d.get('true_op_load_num',0) # number of operatrs with shifts that need to be loaded

        self.static_A_s_path = self.d.get('static_A_s_path') #  path to the sparse matrix with out a shift
        self.load_static_A_s = (self.static_A_s_path is not None)

        self.test_A_s_path = self.d.get('test_A_s_path') # path to the sparse matrix with a shift for testing while training
        self.test_the_model = self.d.get('test_the_model',self.test_A_s_path is not None) # only works for ISTA
        
        # amplitude,frequifenzi,phaseschift
        self.just_one = self.d.get('just_one') # toggle if only training on 1 operator
        if self.d.get('shift_params_path') is not None: # path to saved numy array containing shift parameters
            self.shift_params = np.load(self.d.get('shift_params_path'))
        if self.d.get('shift_params_list_path') is not None: # path to saved list of numpy arrays containing shift parameters
            self.shift_params_list = np.load(self.d.get('shift_params_list_path'))
        else:
            self.shift_params_list=[]
        self.num_fixed = self.d.get('num_fixed') # number of fixed operator that are usd every iteration
        self.num_rand = self.d.get('num_rand') # number of randomly choosen operators each iteration
        self.num_rand_phantoms = self.d.get('num_rand_phantoms') # number of randomly phantoms
        self.rand_phantoms_e_p = self.d.get('rand_phantoms_e_p') # size of the relative noise for the data
        self.num_rand_ops_load = self.d.get('num_rand_ops_load',0)# number of  operators loaded from a list for the iterartion
        self.num_rand_ops_new = self.d.get('num_rand_ops_new',0)#number of new created operators for the iterartion
        self.num_params_shift = self.d.get('num_params_shift',1)# n  sinuses
        self.amp_range = self.d.get('amp_range', [0.03,0.05])# amplitude for shifts
        self.freq_range = self.d.get('freq_range',[500,5000])# frequency ofshifts
        self.freq_shift_range = self.d.get('freq_shift_range',[0,2*np.pi]) # range of shifts
        self.use_true_op_xis = self.d.get('use_true_op_xis',False)# start with 
        
        
        
        self.fw_data_selection_rule = self.d.get('fw_data_selection_rule','All') # selestion rule for training sets
        self.adj_data_selection_rule = self.d.get('adj_data_selection_rule','All')
        self.max_len_dataset = self.d.get('max_len_dataset',5000) #maximal length of the training set
        self.num_train_zeros = self.d.get('num_train_zeros',0)#number of [0,0] elements that are added to the forward and adjoint dataset this is in addition to the before createted datase, i.e. the dataset is the larger then max_len_dataset       
        
        #Model
        self.kernel_size = self.d.get('kernel_size') # Kernel size of the unets
        self.num_of_downs = self.d.get('num_of_downs') # number of contractig and expansive sets
        self.start_channels_power = self.d.get('start_channels_power') # the number we start with befre contrating the first time
        self.last_layer = self.d.get('last_layer') # 'static_ray_trafo' # old was for positioning the unets
        self.forward_model_type = self.d.get('forward_model_type') # selects the used unet 
        self.forward_swaped = self.d.get('forward_swaped') # when true expect the imprecise operator to be in the forward model
        self.learning_rate = self.d.get('learning_rate')#learning rate for the forward model
        
        self.adj_kernel_size = self.d.get('adj_kernel_size') #the same as  for the forward model
        self.adj_num_of_downs = self.d.get('adj_num_of_downs')
        self.adj_model_type = self.d.get('adj_model_type')
        self.adj_swaped = self.d.get('adj_swaped')
        self.adj_start_channels_power = self.d.get('adj_start_channels_power')

        self.overwrite_model_saves = self.d.get('overwrite_model_saves',False) # each iteratiton the saved model gets overwriten
        #loss_fn
        #optimizer
        #sheduler
        self.schedular_patientce = self.d.get('schedular_patientce',50000) #forward model schedular patience
        self.adj_schedular_patientce = self.d.get('adj_schedular_patientce',50000)#adjoint model schedular patience
        self.model_state_dict_path = self.d.get('model_state_dict_path')# path if a model schould be loaded in at the begining for refinement
        self.adjoint_model_state_dict_path = self.d.get('adjoint_model_state_dict_path')
        self.loops_per_iteation = self.d.get('loops_per_iteration',1) #atempt to do more training cycles of forward adjoint model trainin gon one iteration (currently unused)



