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
from utils import fedAvg_gradientNorm_clientSelection, fedAvg_random_clientSelection

# Define parameters
args = parse_args()
args.weight_decay = 0
args.train_batch_size = 16
args.test_batch_size = 64
args.num_epochs = args.iteration_number
args.lambda_reg_arr = 0 #[0, 0.01, 0.1, 1]
args.global_agg = 5
#args.subchannel_number_arr = [5000, 10000, 20000]
#args.cols_arr = [int(x/args.rows) for x in args.subchannel_number_arr]
args.k_arr = np.multiply([2, 5, 10], 10**3)
args.attenuation_factor_arr = [1]



#args.fps_accuracy = np.zeros((args.avg, len(args.lambda_reg_arr), len(args.cols_arr), len(args.k_arr), len(args.attenuation_factor_arr), int(args.num_epochs/args.global_agg)))
#args.fedprox_accuracy = np.zeros((args.avg, len(args.lambda_reg_arr), int(args.num_epochs/args.global_agg)))
#args.randk_accuracy = np.zeros((args.avg,len(args.cols_arr) , int(args.num_epochs/args.global_agg)))
#args.topk_accuracy = np.zeros((args.avg,len(args.cols_arr) , int(args.num_epochs/args.global_agg)))
#args.fetchSGD_accuracy = np.zeros((args.avg, len(args.cols_arr), len(args.k_arr), len(args.attenuation_factor_arr), int(args.num_epochs/args.global_agg)))

args.fedAvg_gradientNorm_CS = np.zeros((args.avg , int(args.num_epochs/args.global_agg)))
args.fedAvg_random_CS = np.zeros((args.avg , int(args.num_epochs/args.global_agg)))


for avg in range(args.avg):
    args.lambda_reg = 0
    args.attenuation_factor = 1 
    
    # Call client selection based on maximum gradient norm
    args.fedAvg_gradientNorm_CS[avg,:] = fedAvg_gradientNorm_clientSelection(args) 

    # Call client selection based on random permutation
    args.fedAvg_random_CS[avg,:] = fedAvg_random_clientSelection(args) 
    

meta_data = {
        "args": args
    }
   
meta_data_file_name = f"MNIST_accuracy_{args.dist}_alpha_{args.alpha}_noise_{args.noise_std}_avg_{args.avg}_epoch_{args.global_agg}_{args.timestamp}.pkl"

simulation_dir = 'Simulations/adaptiveClientSelection/'

# Create the simulation directory if it doesn't exist
if not os.path.exists(simulation_dir):
    os.makedirs(simulation_dir)

meta_data_save_path = os.path.join(simulation_dir, meta_data_file_name)

# Save metadata to file
with open(meta_data_save_path, "wb") as f:
    pickle.dump(meta_data, f)
