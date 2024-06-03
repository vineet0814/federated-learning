# Python methods and libraries
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, Subset
import os
import numpy as np
import pickle
from datetime import datetime
import pickle
import random
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
mpl.rc('font', size=16)#, family='times new roman')
mpl.rc('lines', linewidth=3)
mpl.rc('xtick.major', size=6, width=1.4)
mpl.rc('ytick.major', size=6, width=1.4)
mpl.rc('grid', linewidth=1.4)
mpl.rc('axes.formatter', useoffset=False)
mpl.rc('axes', linewidth=3, xmargin=0)
mpl.rc('savefig', dpi=300, bbox='tight', pad_inches=0.01)
mpl.rc(
    'legend', framealpha=0.8, edgecolor='k', labelspacing=.1,
    borderaxespad=.3, handlelength=.7, handletextpad=.4,
)

# User defined methods and libraries
from config import parse_args, save_module
from model_new import learning_model_reg
from data_loader import load_batch, load_test_batch
from schedule import mask_generator, get_channel_coef
from csvec import CSVec


#Define parameters

args = parse_args()
args.weight_decay = 0
args.num_epochs = args.iteration_number
args.lambda_reg_arr = [0, 0.01, 0.1, 1]
args.global_agg = 5
args.subchannel_number_arr = [5000, 10000, 20000]
args.cols_arr = [int(x/args.rows) for x in args.subchannel_number_arr]
args.k_arr = np.multiply([2, 5, 10], 10**3)
args.attenuation_factor_arr = [1]


        
args.fps_accuracy = np.zeros((args.avg, len(args.lambda_reg_arr), len(args.cols_arr), len(args.k_arr), len(args.attenuation_factor_arr), int(args.num_epochs/args.global_agg)))
args.fedprox_accuracy = np.zeros((args.avg, len(args.lambda_reg_arr), int(args.num_epochs/args.global_agg)))
args.randk_accuracy = np.zeros((args.avg,len(args.cols_arr) , int(args.num_epochs/args.global_agg)))
args.noComp_accuracy = np.zeros((args.avg , int(args.num_epochs/args.global_agg)))
args.topk_accuracy = np.zeros((args.avg,len(args.cols_arr) , int(args.num_epochs/args.global_agg)))
args.fetchSGD_accuracy = np.zeros((args.avg, len(args.cols_arr), len(args.k_arr), len(args.attenuation_factor_arr), int(args.num_epochs/args.global_agg)))


for avg in range(args.avg):
    args.lambda_reg = 0
    args.attenuation_factor = 1 
    args.noComp_accuracy[avg,:] = noComp(args) 
for avg in range(args.avg):
    args.current_avg_round = avg
    for reg in range(len(args.lambda_reg_arr)):
        for ii in range(len(args.cols_arr)):
            for jj in range(len(args.k_arr)):
                for ell in range(len(args.attenuation_factor_arr)):
                    args.lambda_reg = args.lambda_reg_arr[reg]
                    args.cols = args.cols_arr[ii]
                    args.k = args.k_arr[jj]
                    args.attenuation_factor = args.attenuation_factor_arr[ell]
                    args.fps_accuracy[avg, reg, ii,jj,ell,:] = fps(args)
                   
for avg in range(args.avg):
    for reg in range(len(args.lambda_reg_arr)):
        args.lambda_reg = args.lambda_reg_arr[reg]
        args.fedprox_accuracy[avg, reg,:] = fedprox(args)        

for avg in range(args.avg):
    args.current_avg_round = avg
    for ii in range(len(args.cols_arr)):
        for jj in range(len(args.k_arr)):
            for ell in range(len(args.attenuation_factor_arr)):
                    args.lambda_reg = args.lambda_reg_arr[0]
                    args.cols = args.cols_arr[ii]
                    args.k = args.k_arr[jj]
                    args.attenuation_factor = args.attenuation_factor_arr[ell]
                    args.fetchSGD_accuracy[avg, ii,jj,ell,:] = fetchSGD(args)
for avg in range(args.avg):
    args.current_avg_round = avg
    for ii in range(len(args.cols_arr)):
        args.lambda_reg = args.lambda_reg_arr[0]
        args.subchannel_number = args.subchannel_number_arr[ii]
        args.randk_accuracy[avg, ii,:] = randk(args)

         

for avg in range(args.avg):
    args.current_avg_round = avg
    for ii in range(len(args.cols_arr)):
        args.lambda_reg = args.lambda_reg_arr[0]
        args.subchannel_number = args.subchannel_number_arr[ii]
        args.topk_accuracy[avg, ii,:] = topk(args)
for avg in range(args.avg):
    args.lambda_reg = 0
    args.attenuation_factor = 1 
    args.noComp_accuracy[avg,:] = noComp(args) 
        

meta_data = {
        "args": args
    }
if args.noComp_flag:
    meta_data_file_name = f"MNIST_accuracy_noComp_{args.dist}_avg_{args.avg}_epoch_{args.global_agg}_{args.timestamp}.pkl"
else:    
    meta_data_file_name = f"MNIST_accuracy_{args.dist}_alpha_{args.alpha}_noise_{args.noise_std}_avg_{args.avg}_epoch_{args.global_agg}_{args.timestamp}.pkl"

simulation_dir = 'Simulations'    
meta_data_save_path = simulation_dir+"/pkl_files/"+meta_data_file_name
with open(meta_data_save_path, "wb") as f:
    pickle.dump(meta_data, f)
