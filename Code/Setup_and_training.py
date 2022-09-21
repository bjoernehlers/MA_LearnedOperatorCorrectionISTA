    """
        call this skript in Terminla 

        python Setup_and_training.py run_name

        with run_name name of run for tha conf file is already created the training will then start
    Returns:
        none
    """
import numpy as np
import random
import astra
import matplotlib.pyplot as plt
from ray_transforms import get_static_ray_trafo
from select_model_type_matrix import select_fwd_model_type, select_adj_model_type
from Dataset import DataSet,fw_Train_sets,adj_Train_sets
from util import get_op,sparse_Mat,Timer,added_estimate,error_for_y
from training_functions import train_loop,val_loop
from Test_Model import Test,save_plot_Test
from test_model_functions_swaped import get_net_corected_operator
from Discord_bote import sende_den_boten
import torch 
import torch.nn as nn
import odl

from torchinfo import summary
from scipy import rand, sparse

from conf import config
from Iteration_Steps import ISTA_step,GD_step
import os
import sys

# loading the config file
run_name = sys.argv[-1]
print(run_name)
# if run_name == 'setup_and_train_mR_swaped.py':
#     run_name = 'Test_2'
c = config(run_name)

#selecting device for training

device = f"cuda:{c.gpu_idx}" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
# import astra
astra.set_gpu_index(c.gpu_idx)


#operators
static_ray_trafo = get_static_ray_trafo(c.x_res, c.y_res,c.num_angles,c.detector_points,detector_len = c.detector_len)
if c.load_static_A_s:
    static_A_s = sparse.load_npz(c.static_A_s_path)
else:
    static_ray_trafo = get_static_ray_trafo(c.x_res, c.y_res,c.num_angles,c.detector_points)
    static_A_s = sparse_Mat(static_ray_trafo,c.x_res, c.y_res,c.num_angles,c.detector_points)
static_op = get_op(static_A_s,c.x_res, c.y_res, c.num_angles, c.detector_points)

if c.just_one:
    c.shift_params_list = [c.shift_params]


#iteration solver

if c.step_type == 'ISTA':
    step_op = ISTA_step(c.mu,c.lam)
elif c.step_type == 'GD':
    if c.R == 'L2':
        R  = odl.solvers.L2Norm(static_ray_trafo.domain)
        grad_R = R.gradient
    elif c.R == 'Huber':
        R = odl.solvers.Huber(static_ray_trafo.domain,0.001)
        grad_R = R.gradient
    step_op = GD_step(c.mu,c.lam,grad_R)

#Test (only works for ISTA)
def ISTA_obj_func(op,y,lam,x):
    return 1/(2*lam)*np.linalg.norm((op(x)-y).reshape(y.size),2)**2+np.linalg.norm(x.reshape(x.size),1)
def obj_func(op,y,lam,R,x):
    return 1/(2)*np.linalg.norm((op(x)-y).reshape(y.size),2)**2+lam*R(x)
if c.test_the_model:
    test_path = f'{c.path}Test/'
    if not os.path.isdir(test_path):
        os.makedirs(test_path)
    p = np.load(c.test_phantom_path)[0,:,:]
    test_A_s = sparse.load_npz(c.test_A_s_path)
    test_op = get_op(test_A_s,c.x_res, c.y_res, c.num_angles, c.detector_points)
    y = (test_A_s@p.reshape(p.size)).reshape( c.num_angles, c.detector_points)
    test_y_e = error_for_y(y,c.e_p)
    if c.step_type == 'ISTA':
        L = lambda x :ISTA_obj_func(test_op,test_y_e,c.lam,x)
    elif c.step_type == 'GD':
        L = lambda x :obj_func(test_op,test_y_e,c.lam,R,x)
    if c.x_0_selection_rule == '0':
        x_0 = np.zeros((c.x_res,c.y_res))
    elif c.x_0_selection_rule == 'adj':
        x_0 = static_op.adjoint(test_y_e)
    static_Test_dic = Test(step_op,L,test_op,test_y_e,p,x_0,static_op,static_op.adjoint,c.lam,c.mu,100)
    np.save(test_path + f'test_dict_static.npy',static_Test_dic)
    true_Test_dic = Test(step_op,L,test_op,test_y_e,p,x_0,test_op,test_op.adjoint,c.lam,c.mu,100)
    np.save(test_path + f'test_dict_true.npy',true_Test_dic)
    background = {'static':static_Test_dic.get('loss'),
                    'true':true_Test_dic.get('loss')}

def testing(fw_model,adj_model,iteration):
    if not c.test_the_model:
        pass
    else:
        cor_op = get_net_corected_operator(static_op, fw_model,device = device,swaped=c.forward_swaped)
        cor_adj_op = get_net_corected_operator(static_op.adjoint, adj_model,device = device,swaped=c.adj_swaped)
        Test_dic = Test(step_op,L,test_op,test_y_e,p,x_0,cor_op,cor_adj_op,c.lam,c.mu,100)
        save_plot_Test(test_path + f'test_plot_iter_{iteration}.png',Test_dic,background)
        np.save(test_path + f'test_dict_iter_{iteration}.npy',Test_dic)



#DataSet
D = DataSet(c,static_op,device,step_op)

#Models
TA = torch.sparse_csr_tensor(static_A_s.indptr,static_A_s.indices,static_A_s.data)
fw_model = select_fwd_model_type(c,TA.to_dense().to(device)).to(device)

if c.model_state_dict_path is not None:
    fw_model.load_state_dict(torch.load(c.model_state_dict_path))
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(fw_model.parameters(), lr=c.learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=c.schedular_patientce)

s_AT_s = sparse.csr_matrix(static_A_s.T)
TAT = torch.sparse_csr_tensor(s_AT_s.indptr,s_AT_s.indices,s_AT_s.data,s_AT_s.shape)
adj_model = select_adj_model_type(c,TAT.to_dense().to(device)).to(device)
if c.adjoint_model_state_dict_path is not None:
    adj_model.load_state_dict(torch.load(c.adjoint_model_state_dict_path))

adj_loss_fn = nn.MSELoss()
adj_optimizer = torch.optim.Adam(adj_model.parameters(), lr=c.learning_rate)
adj_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(adj_optimizer, 'min',patience=c.adj_schedular_patientce)

summary(fw_model, input_size=(1, 1, c.x_res, c.y_res), depth=4)
summary(adj_model, input_size=(1, 1,c.num_angles , c.detector_points), depth=4)


rand_shift_dict = dict()
rand_shift_dict.update({"num_params":c.num_params_shift})
rand_shift_dict.update({"amp_range":c.amp_range})
rand_shift_dict.update({"freq_range":c.freq_range})
rand_shift_dict.update({"freq_shift_range":c.freq_shift_range})

#training
timer = Timer(c.num_iterations,path=c.path,est_func=added_estimate().est_func)
D.renew_fixed_train_xis_Ys(c.num_fixed)
comp_train_loss = list()
comp_val_loss = list()
adjoint_comp_train_loss = list()
adjoint_comp_val_loss = list()


for i in range(c.start_iteration,c.num_iterations):
    print("#########")
    print(i)
    print("#########")
    #computing dataset for new itration
    D.renew_rand_train_xis_Ops(c.num_rand_ops_load,make_new = False)
    D.renew_rand_train_xis_Ops(c.num_rand_ops_new,make_new = True,add=True)
    D.renew_rand_train_xis_Ys(c.num_rand)
    if c.use_true_op_xis:
        D.update_x_is_true_op(i)
    else:
        D.update_x_is(fw_model,adj_model,i)

    #forward traing
    val_set = D.get_fw_data('val',selection_rule='All')
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1,
                                                    shuffle=False)
    
    num_epochs = c.num_epoch_list[i]
    train_loss = list()
    val_loss = list()
    TrainSets = fw_Train_sets(D)
    for epoch in range(num_epochs):
        train_loader = TrainSets.give_train_loader()
        train_loss.append(train_loop(train_loader,fw_model,loss_fn, optimizer,device))
        val_loss.append(val_loop(val_loader, fw_model, loss_fn,device))
        scheduler.step(val_loss[epoch])

    comp_train_loss = comp_train_loss +  train_loss
    comp_val_loss =comp_val_loss +  val_loss
    
    # adjoint trining

    val_set = D.get_adj_data(fw_model,'val',selection_rule='All')
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1,
                                                    shuffle=False)
    num_epochs = c.num_adj_epoch_list[i]
    train_loss = list()
    val_loss = list()
    TrainSets = adj_Train_sets(D,fw_model)
    for epoch in range(num_epochs):
        train_loader = TrainSets.give_train_loader()
        train_loss.append(train_loop(train_loader,adj_model,adj_loss_fn, adj_optimizer,device))
        val_loss.append(val_loop(val_loader, adj_model, adj_loss_fn,device))
        adj_scheduler.step(val_loss[epoch])

    adjoint_comp_train_loss = adjoint_comp_train_loss + train_loss
    adjoint_comp_val_loss = adjoint_comp_val_loss + val_loss
        


    if c.overwrite_model_saves:
        torch.save(fw_model.state_dict(), f'{c.model_path}model')
        torch.save(adj_model.state_dict(), f'{c.model_path}adjoint_model')
    else:    
        torch.save(fw_model.state_dict(), f'{c.model_path}model_iter_{i}')
        torch.save(adj_model.state_dict(), f'{c.model_path}adjoint_model_iter_{i}')
    
    comp_loss = np.array([np.array(comp_train_loss).flatten(),np.array(comp_val_loss).flatten()])
    adjoint_comp_loss = np.array([np.array(adjoint_comp_train_loss).flatten(),np.array(adjoint_comp_val_loss).flatten()])
        
    np.save(c.path+'fw_loss',comp_loss)
    np.save(c.path+'adjoint_loss',comp_loss)

    fig, axs = plt.subplots(2,1)
    axs[0].plot(comp_loss[0], label = 'train loss')
    axs[0].plot(comp_loss[1], label = 'val loss')
    axs[0].set_yscale('log')
    axs[0].legend()
    axs[1].plot(adjoint_comp_loss[0], label = 'train loss')
    axs[1].plot(adjoint_comp_loss[1], label = 'val loss')
    axs[1].set_yscale('log')
    axs[1].legend()
    axs[0].set_title("model loss")
    axs[1].set_title("adjoint model loss")
    name = "complete_losses"
    fig.tight_layout()
    fig.savefig(f"{c.path}{name}.{c.fileending}")
    if c.doent_show:
        plt.close(fig)
    else:
        plt.show(fig)

    testing(fw_model,adj_model,i)

    timer.make_estimate(i+1)

if not c.overwrite_model_saves:
    torch.save(fw_model.state_dict(), f'{c.model_path}model_last')
    torch.save(adj_model.state_dict(), f'{c.model_path}adjoint_model_last')

comp_loss = np.array([np.array(comp_train_loss).flatten(),np.array(comp_val_loss).flatten()])
adjoint_comp_loss = np.array([np.array(adjoint_comp_train_loss).flatten(),np.array(adjoint_comp_val_loss).flatten()])
    
fig, axs = plt.subplots(2,1)
axs[0].plot(comp_loss[0], label = 'train loss')
axs[0].plot(comp_loss[1], label = 'val loss')
axs[0].set_yscale('log')
axs[0].legend()
axs[1].plot(adjoint_comp_loss[0], label = 'train loss')
axs[1].plot(adjoint_comp_loss[1], label = 'val loss')
axs[1].set_yscale('log')
axs[1].legend()
axs[0].set_title("model loss")
axs[1].set_title("adjoint model loss")
name = "complete_losses"
fig.tight_layout()
fig.savefig(f"{c.path}{name}.{c.fileending}")
fig.savefig(f"{c.path}{name}.png")
if c.doent_show:
    plt.close(fig)
else:
    plt.show(fig)

timer.finished()

images =  []
images.append(f"{c.path}"+"complete_losses.png")
images.append(test_path+f'test_plot_iter_{i}.png')
sende_den_boten(f'Server is fertig auf GPU {c.gpu_idx}',images)
print('fwlr')

    





