# contains a class that creates and containes the trainng data
from logging import lastResort
from sqlite3 import DatabaseError
import numpy as np
import torch
import random
from scipy import sparse
from tqdm import tqdm
import dival
from conf import config
from test_model_functions_swaped import get_net_corected_operator,comb_net_corected_operator
from ray_transforms import get_ray_trafo, get_static_ray_trafo
from util import get_op,sparse_Mat,rand_shift_params,get_shift,error_for_y,im_norm

class Xis():
    #object to store the trainings data wit coordinate  (op,y,x_i)
    def __init__(self) -> None:
        self.xi_List = []
        self.iter_idx = [0,0,0]
    def __getitem__(self,idx):
        if type(idx) is slice:
            return self.xi_List[idx]
        elif type(idx) is int:
            return self.xi_List[idx]
        elif len(idx) == 2:
            return self.xi_List[idx[0]][1][idx[1]]
        elif len(idx) == 3:
            return self.xi_List[idx[0]][1][idx[1]][1][idx[2]]
            
    def __iter__(self):
        self.iter_idx = [0,0,0]
        return self
    def __inbounds__(self,i,j,k):
        return i < len(self.xi_List) and j < len(self.xi_List[i][1]) and k < len(self.xi_List[i][1][j][1])
    def __next__(self):
        [i,j,k] = self.iter_idx
        if self.__inbounds__(i,j,k):
            self.iter_idx = [i,j,k+1]
            return self[i,j,k]
        else:
            k = 0
            j = j + 1
        if self.__inbounds__(i,j,k):
            self.iter_idx = [i,j,k+1]
            return self[i,j,k]
        else:
            j = 0
            i = i+1
        if self.__inbounds__(i,j,k):
            self.iter_idx = [i,j,k+1]
            return self[i,j,k]
        else:
            raise StopIteration
    def append(self,x,op_idx = None,y_idx = None):
        if op_idx is None:
            self.xi_List.append([x,[]])
        elif y_idx is None:
            self.xi_List[op_idx][1].append([x,[]])
        else:
            self.xi_List[op_idx][1][y_idx][1].append(x)

    def index_list(self):
        index_list = list()
        go_on = True
        indx = [0,0,0]
        while go_on:
            [i,j,k] = indx
            if self.__inbounds__(i,j,k):
                indx = [i,j,k+1]
                index_list.append([i,j,k])
            else:
                k = 0
                j = j + 1
                if self.__inbounds__(i,j,k):
                    indx = [i,j,k+1]
                    index_list.append([i,j,k])  
                else:
                    j = 0
                    i = i+1
                    if self.__inbounds__(i,j,k):
                        indx = [i,j,k+1]
                        index_list.append([i,j,k])  
                    else:
                        go_on = False  
        return index_list
            
    def get_list(self,idx,Op=True,Y=True,X=True):
        l = []
        if Op:
            l.append(self[idx[0]][0])
        if Y:
            l.append(self[idx[0],idx[1]][0])
        if X:
            l.append(self[idx[0],idx[1],idx[2]])
        return l

class Xis_OPY_iterator():
    # for looping trugh the operators and y of x_i
    def __init__(self,parent) -> None:
        self.iter_idx = [0,0]
        self.parent = parent
    def __inbounds__(self,i,j):
        return i < len(self.parent.xi_List) and j < len(self.parent.xi_List[i][1]) 
    def __iter__(self):
        self.iter_idx = [0,0]
        return self
    def __next__(self):
        [i,j] = self.iter_idx
        if self.__inbounds__(i,j):
            self.iter_idx = [i,j+1]
            return self.parent[i][0],self.parent[i,j][0],self.parent[i,j][1]
        else:
            j = 0
            i = i+1
        if self.__inbounds__(i,j):
            self.iter_idx = [i,j+1]
            return self.parent[i][0],self.parent[i,j][0],self.parent[i,j][1]
        else:
            raise StopIteration


class Xis_Y_iterator():
    # for looping trugh y of an Xis object
    def __init__(self,parent) -> None:
        self.iter_idx = [0,0]
        self.parent = parent
    def __inbounds__(self,i,j):
        return i < len(self.parent.xi_List) and j < len(self.parent.xi_List[i][1]) 
    def __iter__(self):
        self.iter_idx = [0,0]
        return self
    def __next__(self):
        [i,j] = self.iter_idx
        if self.__inbounds__(i,j):
            self.iter_idx = [i,j+1]
            return self.parent[i,j][0],self.parent[i,j][1]
        else:
            j = 0
            i = i+1
        if self.__inbounds__(i,j):
            self.iter_idx = [i,j+1]
            return self.parent[i,j][0],self.parent[i,j][1]
        else:
            raise StopIteration

class Xis_Op_iterator():
    # for looping trugh the operators  of an Xis object
    def __init__(self,parent) -> None:
        self.iter_idx = 0
        self.parent = parent
    def __inbounds__(self,i):
        return i < len(self.parent.xi_List)
    def __iter__(self):
        self.iter_idx = 0
        return self
    def __next__(self):
        i = self.iter_idx
        if self.__inbounds__(i):
            self.iter_idx = i+1
            return self.parent[i][0],self.parent[i][1]
        else:
            raise StopIteration


class DataSet():
#class for creatingg the dataset

    def __init__(self,conf_object: config,static_op,device,step_op=None) -> None:
        """
    builds the basic sturcture for training usind Xis objects
        Args:
            conf_object (config): conf file
            static_op (_type_): uncorrected operator
            device (_type_): 
            step_op (_type_, optional): operator for getng the nexdt iteraes. Defaults to None.
        """
        self.c = conf_object
        if self.c.just_one:
            self.c.true_op_load_num = 1
        if self.c.load_A_s:
            Op_list = self.load_true_operators_list(self.c.A_s_path,self.c.true_op_load_num)
        else:
            Op_list = self.get_true_operators_list(shiftparams_list = self.c.shift_params_list)
        self.fixed_Op_list = Op_list
        self.fixed_train_x_is = Xis()
        self.build_scelleton(Op_list, x_is=self.fixed_train_x_is)
        self.rand_train_x_is = Xis()
        self.build_scelleton(Op_list,x_is=self.rand_train_x_is)
        self.val_phantoms = np.load(self.c.val_phantom_path)
        self.val_x_is = Xis()
        if self.c.val_op_list is None:
            self.build_scelleton(Op_list,self.val_phantoms,x_is =self.val_x_is)
        else:
            A_s = self.c.val_op_list[0]
            val_op =get_op(A_s,self.c.x_res, self.c.y_res, self.c.num_angles, self.c.detector_points)
            self.build_scelleton([val_op],self.val_phantoms,x_is =self.val_x_is)
        self.static_op = static_op
        self.step_op = step_op
        self.device = device
    def get_train_phantoms(self,num_fixed,num_rand=0,phantom_path=None):
        """_summary_

        Args:
            num_fixed (int): number of fixed phantoms 
            num_rand (int, optional):number of randomphantoms. Defaults to 0.
            phantom_path (_type_, optional): path to the phantoms if they should be loaded and not created. Defaults to None.

        Raises:
            ValueError: if num_fixed+num_rand>length of phantom list in path

        Returns:
            list: list of phantoms
        """
        fixed_phantoms = list()
        rand_phantoms = list()
        if phantom_path is None:
            Phantom_sets =  dival.datasets.EllipsesDataset(
                image_size=self.c.x_res,train_len=num_fixed,validation_len=0,
                test_len=0,fixed_seeds=True)
            for phan in Phantom_sets.generator("train"):
                fixed_phantoms.append(phan.asarray())
            Phantom_sets =  dival.datasets.EllipsesDataset(
                image_size=self.c.x_res,train_len=num_rand,validation_len=0,
                test_len=0,fixed_seeds=True)
            for phan in Phantom_sets.generator("train"):
                rand_phantoms.append(phan.asarray())

        else:
            Ps = np.load(phantom_path)
            length = len(Ps)
            indx = np.arange(length)
            if length < num_fixed+num_rand:
                raise ValueError("num_fixed+num_rand needs to be smaller the length of the phantoms_list")
    
            fix_indx = indx[0:num_fixed]
            rand_indx = indx[num_fixed:]    
            random.shuffle(rand_indx)

            for i in fix_indx:
                fixed_phantoms.append(Ps[i,:,:]) 
            for i in range(num_rand):
                rand_phantoms.append(Ps[i,:,:])
        
        if not (num_rand == 0 or num_fixed == 0):
            return fixed_phantoms, rand_phantoms
        elif num_rand == 0:
            return fixed_phantoms
        elif num_fixed == 0:
            return rand_phantoms

    def build_scelleton(self,Operator_list,phantoms=[],x_is = Xis()):
        # build nested lists for an Xi_
        for op in Operator_list:
            x_is.append(op)
        for op,l in Xis_Op_iterator(x_is):
            for p in phantoms:
                y_e = error_for_y(op(p),self.c.e_p)
                l.append([y_e,[]])

    def renew_fixed_train_xis_Ys(self,num_fixed=None,phantom_path=None):
        """renews the y_i of the fixed Xis object

        Args:
            num_fixed (_type_, optional): number of phantoms after renewing. Defaults to None. then c.numfixed is used
            phantom_path (_type_, optional): path if thay should be loaded. Defaults to None.
        """
        if num_fixed is None:
            num_fixed = self.c.num_fixed
        phantoms = self.get_train_phantoms(num_fixed=num_fixed,phantom_path=phantom_path)
        for op,l in Xis_Op_iterator(self.fixed_train_x_is):
            l.clear()
            for p in phantoms:
                y_e = error_for_y(op(p),self.c.e_p)
                l.append([y_e,[]])

    def renew_rand_train_xis_Ops(self,num_rand_ops=None,make_new = False,add = False):
        """renews training xis in the rand Xis object

        Args:
            num_rand_ops (_type_, optional): . Defaults to None. then c.numrand ops
            make_new (bool, optional): if they schould not be loaded from a list. Defaults to False.
            add (bool, optional): if the new operators schould be added to the alred exiting operators and not replace them. Defaults to False.
        """
        if num_rand_ops is None:
            num_rand_ops = self.c.num_rand_ops
        rand_ops_params_list = list()
        if make_new:
            for i in range(num_rand_ops):
                rand_ops_params_list.append(rand_shift_params(self.c.num_params_shift,
                                                                self.c.amp_range,
                                                                self.c.freq_range,
                                                                self.c.freq_shift_range))
            Op_list = self.get_true_operators_list(shiftparams_list = rand_ops_params_list)
        else:
            idx = np.arange(len(self.fixed_Op_list))
            random.shuffle(idx)
            Op_list = list()
            for i in range(num_rand_ops):
                Op_list.append(self.fixed_Op_list[idx[i]])
        if not add:
            self.rand_train_x_is = Xis()
        # self.build_scelleton(self.fixed_Op_list,x_is=self.rand_train_x_is)
        self.build_scelleton(Op_list,x_is=self.rand_train_x_is)

    def renew_rand_train_xis_Ys(self,num_rand=None,phantom_path=None):
        """renews the y_i of the fixed Xis object

        Args:
            num_rand (_type_, optional): if none then =c-.num_rand. Defaults to None. 
            phantom_path (_type_, optional): if none new phantoms are created. Defaults to None.
        """
        if num_rand is None:
            num_rand = self.c.num_rand
        for op,l in Xis_Op_iterator(self.rand_train_x_is):
            phantoms = self.get_train_phantoms(num_fixed=0,num_rand=num_rand,phantom_path=phantom_path)
            l.clear()
            for p in phantoms:
                y_e = error_for_y(op(p),self.c.e_p)
                l.append([y_e,[]])
    	

    def get_true_operators_list(self,num = 10,shiftparams_list=None,random_order=False,random_shifts=False,random_shift_components=10):
        """ get precise operators (unused)

        Args:
            num (int, optional): number of operators crated. Defaults to 10.
            shiftparams_list (_type_, optional):if none =c.shift_params_list . Defaults to None.
            random_order (bool, optional): rnadom oreder of of the shifts from list . Defaults to False.
            random_shifts (bool, optional): does notihng at the moment. Defaults to False.
            random_shift_components (int, optional): is not used. Defaults to 10.

        Returns:
            _type_: _description_
        """
        if shiftparams_list is None:
            shiftparams_list = self.c.shift_params_list
        true_op_list = list()
        num = int(np.minimum(num,len(shiftparams_list)))
        if not random_shifts and num>0:
            indexes = np.arange(len(shiftparams_list))
            if random_order:
                random.shuffle(indexes)
            for i in range(num):
                j = indexes[i]
                shift = get_shift(shiftparams_list[j])
                RT = get_ray_trafo(self.c.x_res, self.c.y_res, self.c.num_angles, self.c.detector_points,DET_SHIFT=shift,detector_len=self.c.detector_len)
                A_s = sparse_Mat(RT,self.c.x_res, self.c.y_res, self.c.num_angles, self.c.detector_points)
                true_op_list.append(get_op(A_s,self.c.x_res, self.c.y_res, self.c.num_angles, self.c.detector_points))
        return true_op_list
        # random is missing

    def load_true_operators_list(self,path,num = 10):
        """precise operators are loaded from a liste

        Args:
            path (_type_): path to folder with single operators
            num (int, optional): how many schould be loaded (has no safty stop). Defaults to 10.

        Returns:
            _type_: _description_
        """
        Op_list = list()
        for i in tqdm(range(num)):
            A_s = sparse.load_npz(path+f'ray_trafo_{i}.npz')
            Op_list.append(get_op(A_s,self.c.x_res, self.c.y_res, self.c.num_angles, self.c.detector_points))
        return Op_list
        
    def x_0_chooser(self,selection_rule = '0'):
        """ retruns function to choose x_0

        Args:
            selection_rule (str, optional): selts the rule for seletion '0' gives x_0 =0_X and 'adj' gives A_static^*y_e.  Defaults to '0'.

        Returns:
            _type_: fution to select x_0 depending on y_e
        """
        if selection_rule == '0':
            return self.x_0_is_0
        elif selection_rule == 'adj':
            return self.x_0_is_adj
    def x_0_is_0(self,y):
        return np.zeros((self.c.x_res,self.c.y_res))
    def x_0_is_adj(self,y):
        return self.static_op.adjoint(y)
    

    def update_x_is(self,fw_model,adj_model,iteration,step_op = None,num_of_ops = -1,rand_choice=False) -> None:
        """compute new x_is with te trained models

        Args:
            fw_model (_type_): forward model
            adj_model (_type_): adjoint model
            iteration (_type_): maximal i to what we compute the x_i
            step_op (_type_, optional): operator to compute the next iterate. If None self.step_op is used. Defaults to None.
            num_of_ops (int, optional):  not used. Defaults to -1.
            rand_choice (bool, optional): not used. Defaults to False.
        """
        if step_op is None:
            step_op = self.step_op
        cor_op = get_net_corected_operator(self.static_op, fw_model,device = self.device,swaped=self.c.forward_swaped)
        cor_adj_op = get_net_corected_operator(self.static_op.adjoint, adj_model,device = self.device,swaped=self.c.adj_swaped)
        get_x_0 = self.x_0_chooser(self.c.x_0_selection_rule)
        dic = dict()
        dic.update({'num_iter':iteration+1})
        dic.update({'step_op':step_op})
        dic.update({'cor_op':cor_op})
        dic.update({'cor_adj_op':cor_adj_op})
        for y,l in Xis_Y_iterator(self.fixed_train_x_is):
            self.add_line(y,l,get_x_0,**dic)
        for op,l in Xis_Op_iterator(self.fixed_train_x_is):
            for i in range(self.c.num_rand_phantoms):
                l.append(self.add_phantom_line(op,**dic))

        for y,l in Xis_Y_iterator(self.rand_train_x_is):
            self.add_line(y,l,get_x_0,**dic)
        for op,l in Xis_Op_iterator(self.rand_train_x_is):
            for i in range(self.c.num_rand_phantoms):
                l.append(self.add_phantom_line(op,**dic))

        for y,l in Xis_Y_iterator(self.val_x_is):
            self.add_line(y,l,get_x_0,**dic)
        for op,l in Xis_Op_iterator(self.val_x_is):
            for i in range(self.c.num_rand_phantoms):
                l.append(self.add_phantom_line(op,**dic))

    def add_line(self,y,l,get_x_0,num_iter,step_op,cor_op,cor_adj_op):
        #creating a line of x_i from y_e and x_0
        F_abl = lambda x: cor_adj_op(cor_op(x)-y)
        l.clear()
        l.append(get_x_0(y))
        for i in range(1,num_iter):
            l.append(step_op(l[-1],F_abl(l[-1])))
    
    def add_phantom_line(self,true_op,num_iter,step_op,cor_op,cor_adj_op):
        #creating a line of x_i from y_e and x_0=p
        p = self.get_train_phantoms(num_fixed=0,num_rand=1)[0]
        y = true_op(p)
        p_e = error_for_y(p,self.c.rand_phantoms_e_p)
        y_e = error_for_y(y,self.c.e_p)
        l = [p_e]
        F_abl = lambda x: cor_adj_op(cor_op(x)-y)
        for i in range(1,num_iter):
            l.append(step_op(l[-1],F_abl(l[-1])))
        y_l_list = [y_e,l]
        return y_l_list

    def update_x_is_true_op(self,iteration,num_rand_phantoms = None,step_op = None) -> None:
        """computes updates with the tru operators instead of the correted onse

        Args:
            iteration (_type_): number of iterations that schould be computed
            num_rand_phantoms (_type_, optional): If none c.num_rad_phantoms . Defaults to None.
            step_op (_type_, optional): function to compute the next iterarte depending on x and grad Fx If none c.step_op is used. Defaults to None.
        """
        if step_op is None:
            step_op = self.step_op
        if num_rand_phantoms is None:
            num_rand_phantoms = self.c.num_rand_phantoms
        get_x_0 = self.x_0_chooser(self.c.x_0_selection_rule)
        dic = dict()
        dic.update({'num_iter':iteration+1})
        dic.update({'step_op':step_op})
        
        for op,l_y in Xis_Op_iterator(self.fixed_train_x_is):
            dic.update({'cor_op':op})
            dic.update({'cor_adj_op':op.adjoint})
            for y,l in l_y:
                self.add_line(y,l,get_x_0,**dic)
            for i in range(num_rand_phantoms):
                l_y.append(self.add_phantom_line(op,**dic))
            
        for op,l_y in Xis_Op_iterator(self.rand_train_x_is):
            dic.update({'cor_op':op})
            dic.update({'cor_adj_op':op.adjoint})
            for y,l in l_y:
                self.add_line(y,l,get_x_0,**dic)
            for i in range(num_rand_phantoms):
                l_y.append(self.add_phantom_line(op,**dic))
        
        for op,l_y in Xis_Op_iterator(self.val_x_is):
            dic.update({'cor_op':op})
            dic.update({'cor_adj_op':op.adjoint})
            for y,l in l_y:
                self.add_line(y,l,get_x_0,**dic)
            for i in range(num_rand_phantoms):
                l_y.append(self.add_phantom_line(op,**dic))

    def append_fw_data(self,x_is : Xis ,dataset = list(),selection_rule = None):
        """creates forward trining points and ads them to an existin list

        Args:
            x_is (Xis):x_is for generation of the training points
            dataset (_type_, optional): dataset that schould be appended. Defaults to list().
            selection_rule (_type_, optional): selection rule for wich training points schold be contained. Defaults to None.

        Returns:
            _type_: _description_
        """
        if selection_rule is None:
            selection_rule = self.c.fw_data_selection_rule
        
        if selection_rule == 'All':
            for Op,y_e,l in Xis_OPY_iterator(x_is):
                for x_i in l:
                    x = x_i
                    if not self.c.forward_swaped:
                        x = self.static_op(x)
                    z = Op(x_i)
                    dataset.append([torch.from_numpy(x),torch.from_numpy(z)])

        elif selection_rule.startswith('random_max_'):
            index_list = x_is.index_list()
            idx = np.arange(len(index_list))
            random.shuffle(idx)
            max_len = int(selection_rule.split('random_max_')[-1])
            for i in range(min(len(idx),max_len)):
                Op,x_i = x_is.get_list(index_list[idx[i]],Y=False)
                x = x_i
                if not self.c.forward_swaped:
                    x = self.static_op(x)
                z = Op(x_i)
                dataset.append([torch.from_numpy(x),torch.from_numpy(z)])


        elif selection_rule.startswith('random_perzent_min_'):
            min_len,p = selection_rule.split('random_perzent_min_')[-1].split('_')
            min_len = int(min_len)
            p = float(p)
            index_list = x_is.index_list()
            idx = np.arange(len(index_list))
            random.shuffle(idx)
            for i in range(max(len(idx)*p,min_len)):
                Op,x_i = x_is.get_list(index_list[idx[i]],Y=False)
                x = x_i
                if not self.c.forward_swaped:
                    x = self.static_op(x)
                z = Op(x_i)
                dataset.append([torch.from_numpy(x),torch.from_numpy(z)])

        elif selection_rule.startswith('last_and_random'):
            if not dataset and selection_rule.split('last_and_random')[-1] != '_comb':
                dataset = [[],[]]
            last_dataset = list()
            rest_dataset = list()
            if self.c.forward_swaped:
                a = lambda x: x
            else:
                a = lambda x: self.static_op(x)

            for Op,y_e,l in Xis_OPY_iterator(x_is):
                x = a(l[-1])
                z = Op(l[-1])
                last_dataset.append([torch.from_numpy(x),torch.from_numpy(z)])
                for x_i in l[:-1]:
                    x = a(x_i)
                    z = Op(x_i)
                    rest_dataset.append([torch.from_numpy(x),torch.from_numpy(z)])
            if selection_rule.split('last_and_random')[-1] == '_comb':
                dataset = last_dataset + rest_dataset
            else:
                dataset[0] = dataset[0] + last_dataset
                dataset[1] = dataset[1] + rest_dataset

        elif selection_rule.startswith('random_perzent_min_'):
            min_len,p = selection_rule.split('random_perzent_min_')[-1].split('_')
            min_len = int(min_len)
            p = float(p)
            index_list = x_is.index_list()
            idx = np.arange(len(index_list))
            random.shuffle(idx)
            for i in range(max(len(idx)*p,min_len)):
                Op,x_i = x_is.get_list(index_list[idx[i]],Y=False)
                x = x_i
                if not self.c.forward_swaped:
                    x = self.static_op(x)
                z = Op(x_i)
                dataset.append([torch.from_numpy(x),torch.from_numpy(z)])
        return dataset    

    def get_fw_data(self,data_name:str,selection_rule = None):
        """gives back a datset for training the forward model

        Args:
            data_name (str): trian or val
            selection_rule (_type_, optional): rule for which trining points to select. If None c.fw_selecton _ule is used.  Defaults to None. 

        Returns:
            list: forward trining data iin a list
        """
        data_set = list()
        if data_name == 'val':
            data_set = self.append_fw_data(self.val_x_is,data_set,selection_rule)
        elif data_name == 'train':
            data_set = self.append_fw_data(self.fixed_train_x_is,data_set,selection_rule)
            data_set = self.append_fw_data(self.rand_train_x_is,data_set,selection_rule)
        return data_set

    def append_adj_data(self,fw_model,x_is : Xis ,dataset = list(),selection_rule = None):
        """apped a list with trining poins for the adjoint model

        Args:
            fw_model (_type_): forward model
            x_is (Xis): X_is to compute the training points
            dataset (_type_, optional): list which is to be appended. Defaults to list().
            selection_rule (_type_, optional): selection rule for wich training points schold be contained. Defaults to None.

        Returns:
            _type_: list with new appended traning sets
        """
        if selection_rule is None:
            selection_rule = self.c.fw_data_selection_rule
        cor_op = get_net_corected_operator(self.static_op, fw_model,device = self.device,swaped=self.c.forward_swaped)
        
        if selection_rule == 'All':
            for Op,y_e,l in Xis_OPY_iterator(x_is):
                for x_i in l:
                    r = cor_op(x_i)-y_e
                    if not self.c.adj_swaped:
                        x = self.static_op.adjoint(r)
                    else:
                        x = r
                    z = Op.adjoint(r)
                    dataset.append([torch.from_numpy(x),torch.from_numpy(z)])

        elif selection_rule.startswith('random_max_'):
            index_list = x_is.index_list()
            idx = np.arange(len(index_list))
            random.shuffle(idx)
            max_len = int(selection_rule.split('random_max_')[-1])
            for i in range(min(len(idx),max_len)):
                Op,y_e,x_i = x_is.get_list(index_list[idx[i]])
                r = cor_op(x_i)-y_e
                if not self.c.adj_swaped:
                    x = self.static_op.adjoint(r)
                z = Op.adjoint(r)
                dataset.append([torch.from_numpy(x),torch.from_numpy(z)])


        elif selection_rule.startswith('random_perzent_min_'):
            min_len,p = selection_rule.split('random_perzent_min_')[-1].split('_')
            min_len = int(min_len)
            p = float(p)
            index_list = x_is.index_list()
            idx = np.arange(len(index_list))
            random.shuffle(idx)
            for i in range(max(len(idx)*p,min_len)):
                Op,y_e,x_i = x_is.get_list(index_list[idx[i]])
                r = cor_op(x_i)-y_e
                if not self.c.adj_swaped:
                    x = self.static_op.adjoint(r)
                z = Op.adjoint(r)
                dataset.append([torch.from_numpy(x),torch.from_numpy(z)])

        elif selection_rule.startswith('last_and_random'):
            if not dataset and selection_rule.split('last_and_random')[-1] != '_comb':
                dataset = [[],[]]
            last_dataset = list()
            rest_dataset = list()
            if self.c.adj_swaped:
                a = lambda x: x
            else:
                a = lambda x: self.static_op.adjoint(x)
            for Op,y_e,l in Xis_OPY_iterator(x_is):
                r = cor_op(l[-1])-y_e
                x = a(r)
                z = Op.adjoint(r)
                last_dataset.append([torch.from_numpy(x),torch.from_numpy(z)])
                for x_i in l[:-1]:
                    r = cor_op(x_i)-y_e
                    x = a(r)
                    z = Op.adjoint(r)
                    rest_dataset.append([torch.from_numpy(x),torch.from_numpy(z)])
            if selection_rule.split('last_and_random')[-1] == '_comb':
                dataset = last_dataset + rest_dataset
            else:
                dataset[0] = dataset[0] + last_dataset
                dataset[1] = dataset[1] + rest_dataset

        return dataset  
    
    def get_adj_data(self,fw_model,data_name:str,selection_rule = None):
        """gives back a datset for training the adjoint model

        Args:
            data_name (str): trian or val
            selection_rule (_type_, optional): rule for which trining points to select. If None c.adj_selecton _ule is used.  Defaults to None. 

        Returns:
            list: adjoint trining data in a list
        """
        data_set = list()
        if data_name == 'val':
            data_set = self.append_adj_data(fw_model,self.val_x_is,data_set,selection_rule)
        elif data_name == 'train':
            data_set = self.append_adj_data(fw_model,self.fixed_train_x_is,data_set,selection_rule)
            data_set = self.append_adj_data(fw_model,self.rand_train_x_is,data_set,selection_rule)
        return data_set


    def update_x_is_comb(self,comb_model,iteration,step_op = None) -> None:
        if step_op is None:
            step_op = self.step_op
        cor_comb_op = comb_net_corected_operator(self.static_op, comb_model,device = self.device).get_impl()
        get_x_0 = self.x_0_chooser(self.c.x_0_selection_rule)
        for y,l in Xis_Y_iterator(self.fixed_train_x_is):
            F_abl = lambda x: cor_comb_op(x,y)
            l.clear()
            l.append(get_x_0(y))
            for i in range(1,iteration+1):
                l.append(step_op(l[-1],F_abl(l[-1])))

        for y,l in Xis_Y_iterator(self.rand_train_x_is):
            F_abl = lambda x: cor_comb_op(x,y)
            l.clear()
            l.append(get_x_0(y))
            for i in range(1,iteration+1):
                l.append(step_op(l[-1],F_abl(l[-1])))

        for y,l in Xis_Y_iterator(self.val_x_is):
            F_abl = lambda x: cor_comb_op(x,y)
            l.clear()
            l.append(get_x_0(y))
            for i in range(1,iteration+1):
                l.append(step_op(l[-1],F_abl(l[-1])))

    def append_comb_data(self,x_is : Xis, dataset = list(),selection_rule = None):
        if selection_rule is None:
            selection_rule = self.c.fw_data_selection_rule
        
        if selection_rule == 'All':
            for Op,y_e,l in Xis_OPY_iterator(x_is):
                for x_i in l:
                    r = [torch.from_numpy(x_i),torch.from_numpy(y_e)]
                    z = Op.adjoint(Op(x_i)-y_e)
                    dataset.append([r,torch.from_numpy(z)])

        elif selection_rule.startswith('last_and_random'):
            if not dataset and selection_rule.split('last_and_random')[-1] != '_comb':
                dataset = [[],[]]
            last_dataset = list()
            rest_dataset = list()
            for Op,y_e,l in Xis_OPY_iterator(x_is):
                r = [torch.from_numpy(l[-1]),torch.from_numpy(y_e)]
                z = Op.adjoint(Op(l[-1])-y_e)
                last_dataset.append([r,torch.from_numpy(z)])
                for x_i in l[:-1]:
                    r = [torch.from_numpy(x_i),torch.from_numpy(y_e)]
                    z = Op.adjoint(Op(x_i)-y_e)
                    rest_dataset.append([r,torch.from_numpy(z)])
            if selection_rule.split('last_and_random')[-1] == '_comb':
                dataset = last_dataset + rest_dataset
            else:
                dataset[0] = dataset[0] + last_dataset
                dataset[1] = dataset[1] + rest_dataset
        return dataset  

    def get_comb_data(self,data_name:str,selection_rule = None):
        data_set = list()
        if data_name == 'val':
            data_set = self.append_comb_data(self.val_x_is,data_set,selection_rule)
        elif data_name == 'train':
            data_set = self.append_comb_data(self.fixed_train_x_is,data_set,selection_rule)
            data_set = self.append_comb_data(self.rand_train_x_is,data_set,selection_rule)
        return data_set
    # def get_fw_dataset(self,):
    #     pass

    # def residual_check(self,L=None):
    #     if L is None:
    #         if c.step_type == 'ISTA':
    #             def ISTA_obj_func(op,y,lam,x):
    #                 return 1/(2*lam)*np.linalg.norm((op(x)-y).reshape(y.size),2)**2+np.linalg.norm(x.reshape(x.size),1)
    #             L = lambda x :ISTA_obj_func(test_op,test_y_e,c.lam,x)
    #         elif c.step_type == 'GD':
    #             def obj_func(op,y,lam,R,x):
    #                 return 1/(2)*np.linalg.norm((op(x)-y).reshape(y.size),2)**2+lam*R(x)
    #             L = lambda x :obj_func(test_op,test_y_e,c.lam,R,x)
    #         residuals = list()
    #         for op,y,l in Xis_OPY_iterator(self.fixed_train_x_is):
    #             residu = list()
    #             for x in l:
    #                 residu.append(im_norm(op(x)-y))
    #             resduals.append(residu)
    #         for op,y,l in Xis_OPY_iterator(self.rand_train_x_is):
    #             residu = list()
    #             for x in l:
    #                 residu.append(im_norm(op(x)-y))
    #             resduals.append(residu)
    #         for op,y,l in Xis_OPY_iterator(self.val_x_is):
    #             residu = list()
    #             for x in l:
    #                 residu.append(im_norm(op(x)-y))
    #             resduals.append(residu)
    #     return residuals()
        

class fw_Train_sets():
    # class that stores the dataset and can give back rando smples acording to seletion rule ready for forward training
    def __init__(self,data_set:DataSet, max_len_dataset = None):
        self.D = data_set
        self.c = data_set.c
        if max_len_dataset is None:
            self.max_len_dataset = self.c.max_len_dataset
        else:self.max_len_dataset = max_len_dataset

        self.last_iter_data,self.rest_data  = self.D.get_fw_data('train',selection_rule='last_and_random')

    def give_train_loader(self):
        """returns dataloader for foreward model training made of dataset in according to the train_set_selection_rule in config in the DataSet

        Returns:
            torch.utils.data.dataloader.DataLoader : dataloader made of dataset in according to the train_set_selection_rule in config in the DataSet
        """
        train_set = list()
        if self.c.train_set_selection_rule == 'All':
            complete_train_set = self.last_iter_data+ self.rest_data
            num_rand_data = min(self.max_len_dataset,len(complete_train_set))
            train_set = random.sample(complete_train_set,num_rand_data)
        elif self.c.train_set_selection_rule == 'last_and_random':
            num_rand_data = min(len(self.rest_data),max(self.max_len_dataset-len(self.last_iter_data),0))
            train_set = self.last_iter_data + random.sample(self.rest_data,num_rand_data)
        for i in range(self.c.num_train_zeros):
            train_set.append([np.zeros_like(train_set[0][0]),np.zeros_like(train_set[0][1])])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                                    shuffle=True)
        return train_loader

class adj_Train_sets():
    # class that stores the dataset and can give back rando smples acording to seletion rule ready for adjoint training
    def __init__(self,data_set:DataSet,fw_model, max_len_dataset = None):
        self.D = data_set
        self.c = data_set.c
        if max_len_dataset is None:
            self.max_len_dataset = self.c.max_len_dataset
        self.last_iter_data,self.rest_data  = self.D.get_adj_data(fw_model,'train',selection_rule='last_and_random')

    def give_train_loader(self):
        """returns dataloader for adjoint model training made of dataset in according to the train_set_selection_rule in config in the DataSet
        
        Args:
            fw_model (pytorch model): the alreday trained forward model

        Returns:
            torch.utils.data.dataloader.DataLoader : 
        """
        train_set = list()
        if self.c.train_set_selection_rule == 'All':
            complete_train_set = self.last_iter_data+ self.rest_data
            num_rand_data = min(self.max_len_dataset,len(complete_train_set))
            train_set = random.sample(complete_train_set,num_rand_data)
        elif self.c.train_set_selection_rule == 'last_and_random':
            num_rand_data = min(len(self.rest_data),max(self.max_len_dataset-len(self.last_iter_data),0))
            train_set = self.last_iter_data + random.sample(self.rest_data,num_rand_data)
        for i in range(self.c.num_train_zeros):
            train_set.append([np.zeros_like(train_set[0][0]),np.zeros_like(train_set[0][1])])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                                    shuffle=True)
        return train_loader

