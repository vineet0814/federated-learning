# Python methods and libraries
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, Subset
import os
import numpy as np
import matplotlib as mpl
import pickle
from datetime import datetime
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
import random
import pickle

# User defined methods and libraries
from config import parse_args, save_module
from model_new import learning_model_reg
from data_loader import load_batch, load_test_batch
from schedule import mask_generator, get_channel_coef
from csvec import CSVec
from data_loader_mnist import get_data_loader, get_next_batch



def signalPower(x):
    return np.mean(x**2)
    
def SNR(signal, noise):
    powS = signalPower(signal)
    powN = signalPower(noise)
    return 10*np.log10((powS-powN)/powN)

def gradients_to_vector(gradients):
    """
    Convert a list of gradients (tensors) into a single column vector.

    Args:
        gradients (list of torch.Tensor): List of gradients to be flattened.

    Returns:
        torch.Tensor: A flattened column vector containing all gradients.
    """
    return torch.cat([grad.view(-1) for grad in gradients], dim=0)


def vector_to_gradients(vector, model):
    """
    Convert a single column vector back into gradients of the same shape as the model's parameters.

    Args:
        vector (torch.Tensor): A single column vector containing gradients.
        model (nn.Module): The neural network model whose parameters' shapes are used for reshaping.

    Returns:
        list of torch.Tensor: List of gradients reshaped to match the model's parameters.
    """
    gradients = []
    start_idx = 0
    for param in model.parameters():
        param_size = torch.prod(torch.tensor(param.size()))
        grad_size = param.grad.view(-1).size(0)
        gradients.append(vector[start_idx:start_idx + grad_size].view(param.size()))
        start_idx += grad_size
    return gradients

# Example usage:
# Assuming you have computed gradients and stored them in 'gradients' as a list

# Convert gradients to a single column vector
#vectorized_gradients = gradients_to_vector(gradients)

# Perform operations on the vectorized gradients (e.g., update weights manually)

# Convert the single column vector back to gradients matching the model's parameters
#restored_gradients = vector_to_gradients(vectorized_gradients, your_model)

def update_model_with_vectorized_weights(model, vectorized_weights):
    """
    Update a PyTorch model's parameters using a vectorized representation of weights.

    Args:
        model (nn.Module): The PyTorch model to update.
        vectorized_weights (torch.Tensor): A single column vector containing the weights.

    Returns:
        None: The function updates the model parameters in-place.
    """
    start_idx = 0
    for param in model.parameters():
        param_size = torch.prod(torch.tensor(param.size()))
        new_weights = vectorized_weights[start_idx:start_idx + param_size].view(param.size())
        param.data.copy_(new_weights)
        start_idx += param_size

# Example usage:
# Assuming you have a model 'your_model' and a vectorized representation of weights 'vectorized_weights'

# Update the model parameters with the vectorized weights
#update_model_with_vectorized_weights(your_model, vectorized_weights)


def attenuate_array(arr, k, attenuation_factor):
    '''
        sorted_indices = np.argsort(-np.abs(arr))
        result = arr.copy()

        for i in range(k, len(arr)):
            result[sorted_indices[i]] *= attenuation_factor

        return result
    '''
    # Calculate the indices that correspond to the top-k coordinates
    sorted_indices = np.argpartition(-np.abs(arr), k)
    # Create a copy of the array to avoid modifying the original array
    result = arr.copy()

    # Attenuate the magnitude of coordinates beyond the top-k
    result[sorted_indices[k:]] *= attenuation_factor

    return result
    
def topk(args):
    networks = [learning_model_reg(args) for _ in range(10)]
    reference_params = networks[0].DNN_model.parameters()
    user_grad, user_final_shape, user_initial_shape = networks[0].grad_compute_loader_reg(data_loaders[0],  networks[0].DNN_model, reference_params)
    args.p = len(user_grad)
    topk_accuracy  = []
    remainder = np.zeros((args.p, args.user_number))
    args.local_grad_topk = np.zeros((args.user_number, args.num_epochs, args.p))
    args.agg_grad_topk = np.zeros((args.num_epochs, args.p))
    for epoch in range(args.num_epochs):
        u_m = remainder
        agg_grad = np.zeros(args.p)
        estimator = agg_grad
        y = np.zeros(args.subchannel_number)
        for i, network in enumerate(networks):
            # Compute local gradient
            user_grad, user_final_shape, user_initial_shape = networks[i].grad_compute_loader_reg(data_loaders[i], networks[i].DNN_model, reference_params)
            user_grad = np.multiply(args.learning_rate, user_grad)  + remainder[:,i]
            u_m[:,i] = user_grad
            args.local_grad_topk[i, epoch, :] = user_grad
            agg_grad += user_grad
        mask_indices = np.argpartition(abs(agg_grad), -args.subchannel_number)[-args.subchannel_number:]  # Indices not sorted
        args.agg_grad_topk[epoch, :] = (agg_grad + np.random.normal(args.noise_mean, args.noise_std, args.p))/args.user_number
        y = agg_grad[mask_indices]
        for m in range(args.user_number):
            masked_grads = np.zeros(args.p)
            masked_grads[mask_indices] = u_m[mask_indices,m]
            remainder[:,m] = u_m[:,m] - masked_grads
        noise = np.random.normal(args.noise_mean, args.noise_std, args.subchannel_number)
        y += noise
        y = y/args.user_number
        estimator[mask_indices] = y
        for i, network in enumerate(networks):    
            networks[i].update_params(estimator)    
        batch_test, labels_test = get_next_batch(test_loader)
        accuracy = networks[0].check_accuracy_torch(batch_test, labels_test, epoch)    
        if epoch % args.global_agg == 0 :
            topk_accuracy.append(accuracy)
            print('top-k accuracy at iteration:', epoch,'is: ', accuracy)
    return topk_accuracy
    
def randk(args):
    networks = [learning_model_reg(args) for _ in range(10)]
    reference_params = networks[0].DNN_model.parameters()
    user_grad, user_final_shape, user_initial_shape = networks[0].grad_compute_loader_reg(data_loaders[0],  networks[0].DNN_model, reference_params)
    args.p = len(user_grad)
    randk_accuracy  = []
    args.local_grad_randk = np.zeros((args.user_number, args.num_epochs, args.p))
    args.agg_grad_randk = np.zeros((args.num_epochs, args.p))
    mask_obj = mask_generator(args.subchannel_number, args.p)
    remainder = np.zeros((args.p, args.user_number))
    for epoch in range(args.num_epochs):
        mask, mask_indices = mask_obj.uniform_next()
        u_m = np.zeros(args.subchannel_number)
        agg_grad = np.zeros(args.p)
        for i, network in enumerate(networks):
            # Compute local gradient
            user_grad, user_final_shape, user_initial_shape = networks[i].grad_compute_loader_reg(data_loaders[i], networks[i].DNN_model, reference_params)
            user_grad = np.multiply(args.learning_rate, user_grad) + remainder[:,i]
            args.local_grad_randk[i, epoch, :] = user_grad
            agg_grad += user_grad
            masked_grads = np.multiply(mask, user_grad)
            remainder[:,i] = user_grad - masked_grads
            u_m += masked_grads[mask_indices]
        noise = np.random.normal(args.noise_mean, args.noise_std, args.subchannel_number)
        u_m += noise
        u_m = u_m/args.user_number
        estimator = np.zeros(args.p)
        estimator[mask_indices] = u_m
        args.agg_grad_randk[epoch, :] = (agg_grad + np.random.normal(args.noise_mean, args.noise_std, args.p))/args.user_number
        for i, network in enumerate(networks):    
            networks[i].update_params(estimator)    
        batch_test, labels_test = get_next_batch(test_loader)
        accuracy = networks[0].check_accuracy_torch(batch_test, labels_test, epoch)    
        
        if epoch % args.global_agg == 0 :
            randk_accuracy.append(accuracy)
            print('rand-k accuracy at iteration:', epoch,'is: ', accuracy)
    return randk_accuracy

def fedAvg(args):
    networks = [learning_model_reg(args) for _ in range(10)]
    reference_params = networks[0].DNN_model.parameters()
    user_grad, user_final_shape, user_initial_shape = networks[0].grad_compute_loader_reg(data_loaders[0],  networks[0].DNN_model, reference_params)
    args.p = len(user_grad)
    fedAvg_accuracy  = []
    args.local_grad = np.zeros((args.user_number, args.num_epochs, args.p))
    args.agg_grad = np.zeros((args.num_epochs, args.p))
    for epoch in range(args.num_epochs):
        u_m = np.zeros(args.p)
        for i, network in enumerate(networks):
            # Compute local gradient
            user_grad, user_final_shape, user_initial_shape = networks[i].grad_compute_loader_reg(data_loaders[i], networks[i].DNN_model, reference_params)
            user_grad = np.multiply(args.learning_rate, user_grad) #+ remainder[:,i]
            args.local_grad[i, epoch, :] = user_grad
            u_m += user_grad
        noise = np.random.normal(args.noise_mean, args.noise_std, args.p)
        u_m += noise
        u_m = u_m/args.user_number
        args.agg_grad[epoch, :] = u_m 
        estimator = np.zeros(args.p)
        estimator = u_m
        for i, network in enumerate(networks):    
            networks[i].update_params(estimator)    
        batch_test, labels_test = get_next_batch(test_loader)
        accuracy = networks[0].check_accuracy_torch(batch_test, labels_test, epoch)    
        
        if epoch % args.global_agg == 0 :
            fedAvg_accuracy.append(accuracy)
            print('FedAvg accuracy at iteration:', epoch,'is: ', accuracy)
    return fedAvg_accuracy


def fedAvg_gradientNorm_clientSelection(args):
    networks = [learning_model_reg(args) for _ in range(args.user_number)]
    data_loaders, test_loader = get_data_loader(args)
    reference_params = networks[0].DNN_model.parameters()
    user_grad, user_final_shape, user_initial_shape = networks[0].grad_compute_loader_reg(data_loaders[0],  networks[0].DNN_model, reference_params)
    args.p = len(user_grad)
    fedAvg_CS_accuracy  = []
    args.local_grad = np.zeros((args.user_number, args.num_epochs, args.p))
    args.agg_grad = np.zeros((args.num_epochs, args.p))
    local_gradient_norms = np.zeros(args.user_number)
    for epoch in range(args.num_epochs):
        u_m = np.zeros(args.p)
        # Client selection rule 
        if epoch == 0: 
            chosen_indices = np.random.choice(np.arange(args.user_number), args.num_clients, replace=False)
        elif epoch != 0:
            for u in range(args.user_number):  
                local_gradient_norms[u] += np.linalg.norm(args.local_grad[u, epoch-1, :].flatten())
                
        # Update to central server            
        if epoch % args.global_agg == 0:        
             # Compute local gradient       
            for i, network in enumerate(networks):
                user_grad, user_final_shape, user_initial_shape = networks[i].grad_compute_loader_reg(data_loaders[i], networks[i].DNN_model, reference_params)
                user_grad = np.multiply(args.learning_rate, user_grad) 
                args.local_grad[i, epoch, :] = user_grad
            noise = np.random.normal(args.noise_mean, args.noise_std, args.p)
            if epoch != 0: 
                sorted_indices = np.argsort(local_gradient_norms)[::-1]
                chosen_indices = sorted_indices[:args.num_clients]
            for index in chosen_indices:
                u_m +=  args.local_grad[index, epoch, :]   
            u_m += noise
            u_m = u_m/args.num_clients
            args.agg_grad[epoch, :] = u_m 
            for index in chosen_indices:    
                networks[index].update_params(u_m)                 
            # Re-initialize gradient norm array to zero for next round and compute accuracy    
            local_gradient_norms = np.zeros(args.user_number)    
            batch_test, labels_test = get_next_batch(test_loader)
            accuracy = networks[0].check_accuracy_torch(batch_test, labels_test, epoch)    
            fedAvg_CS_accuracy.append(accuracy)
            print('FedAvg gradient norm based client selection accuracy at iteration:', epoch,'is: ', accuracy)
        else: 
            for i, network in enumerate(networks):
                # Compute local gradient
                user_grad, user_final_shape, user_initial_shape = networks[i].grad_compute_loader_reg(data_loaders[i], networks[i].DNN_model, reference_params)
                user_grad = np.multiply(args.learning_rate, user_grad)
                args.local_grad[i, epoch,: ] = user_grad
                networks[i].update_params(user_grad)
    return fedAvg_CS_accuracy


def fedAvg_random_clientSelection(args):
    networks = [learning_model_reg(args) for _ in range(args.user_number)]
    data_loaders, test_loader = get_data_loader(args)
    reference_params = networks[0].DNN_model.parameters()
    user_grad, user_final_shape, user_initial_shape = networks[0].grad_compute_loader_reg(data_loaders[0],  networks[0].DNN_model, reference_params)
    args.p = len(user_grad)
    fedAvg_random_CS_accuracy  = []
    args.local_grad = np.zeros((args.user_number, args.num_epochs, args.p))
    args.agg_grad = np.zeros((args.num_epochs, args.p))
    for epoch in range(args.num_epochs):
        u_m = np.zeros(args.p)
        # Client selection rule 
        chosen_indices = np.random.choice(np.arange(args.user_number), args.num_clients, replace=False)        
        # Update to central server            
        if epoch % args.global_agg == 0:        
             # Compute local gradient       
            for i, network in enumerate(networks):
                user_grad, user_final_shape, user_initial_shape = networks[i].grad_compute_loader_reg(data_loaders[i], networks[i].DNN_model, reference_params)
                user_grad = np.multiply(args.learning_rate, user_grad) 
                args.local_grad[i, epoch, :] = user_grad
            noise = np.random.normal(args.noise_mean, args.noise_std, args.p)
            for index in chosen_indices:
                u_m +=  args.local_grad[index, epoch, :]   
            u_m += noise
            u_m = u_m/args.num_clients
            args.agg_grad[epoch, :] = u_m 
            for index in chosen_indices:    
                networks[index].update_params(u_m)                 
            # Compute accuracy        
            batch_test, labels_test = get_next_batch(test_loader)
            accuracy = networks[0].check_accuracy_torch(batch_test, labels_test, epoch)    
            fedAvg_random_CS_accuracy.append(accuracy)
            print('FedAvg gradient norm based client selection accuracy at iteration:', epoch,'is: ', accuracy)
        else: 
            for i, network in enumerate(networks):
                # Compute local gradient
                user_grad, user_final_shape, user_initial_shape = networks[i].grad_compute_loader_reg(data_loaders[i], networks[i].DNN_model, reference_params)
                user_grad = np.multiply(args.learning_rate, user_grad)
                args.local_grad[i, epoch,: ] = user_grad
                networks[i].update_params(user_grad)
    return fedAvg_random_CS_accuracy

def fedprox(args):
    networks = [learning_model_reg(args) for _ in range(10)]
    parameter_network = learning_model_reg(args)
    reference_params = networks[0].DNN_model.parameters()
    user_grad, user_final_shape, user_initial_shape = networks[0].grad_compute_loader_reg(data_loaders[0],  networks[0].DNN_model, reference_params)
    args.p = len(user_grad)
    args.local_grad_fedprox = np.zeros((args.user_number, args.num_epochs, args.p))
    args.agg_grad_fedprox = np.zeros((args.num_epochs//args.global_agg, args.p))
    fedprox_accuracy  = []        
    for epoch in range(args.num_epochs):
        u_m = np.zeros(args.p)
        if epoch % args.global_agg == 0:  
            for i, network in enumerate(networks):
                # Compute local gradient
                user_grad, user_final_shape, user_initial_shape = networks[i].grad_compute_loader_reg(data_loaders[i], networks[i].DNN_model, reference_params)
                args.local_grad_fedprox[i, epoch,: ] = np.multiply(args.learning_rate, user_grad)
                u_m += np.multiply(args.learning_rate, user_grad)
            noise = np.random.normal(args.noise_mean, args.noise_std, args.p)
            u_m += noise
            u_m = u_m/args.user_number
            args.agg_grad_fedprox[epoch//args.global_agg, : ] = u_m
            for i, network in enumerate(networks):    
                networks[i].update_params(u_m)    
            batch_test, labels_test = get_next_batch(test_loader)
            accuracy = networks[0].check_accuracy_torch(batch_test, labels_test, epoch)
            reference_params = networks[0].DNN_model.parameters()
            print('Fedprox accuracy at iteration:', epoch,'is: ', accuracy)
            fedprox_accuracy.append(accuracy)
        else:
            for i, network in enumerate(networks):
                # Compute local gradient
                user_grad, user_final_shape, user_initial_shape = networks[i].grad_compute_loader_reg(data_loaders[i], networks[i].DNN_model, reference_params)
                user_grad = np.multiply(args.learning_rate, user_grad)
                args.local_grad_fedprox[i, epoch,: ] = user_grad
                networks[i].update_params(user_grad)
    return fedprox_accuracy

def fetchSGD(args):
    networks = [learning_model_reg(args) for _ in range(10)]
    parameter_network = learning_model_reg(args)
    reference_params = networks[0].DNN_model.parameters()
    user_grad, user_final_shape, user_initial_shape = networks[0].grad_compute_loader_reg(data_loaders[0],  networks[0].DNN_model, reference_params)
    args.p = len(user_grad)    
    fetchSGD_accuracy  = []
    theta = np.zeros(args.p)
    args.local_grad_fsgd = np.zeros((args.user_number, args.num_epochs, args.p))
    args.agg_grad_fsgd = np.zeros((args.num_epochs, args.p))
    theta_local = np.zeros((args.user_number, args.p))
    agg_grad = np.zeros(args.p)
    #initialize a residual sketch, this will be used to replicate all other sketches
    S_e = CSVec(args.p, args.cols, args.rows, doInitialize=True, device='cpu', numBlocks=1)
    S_e.table = torch.zeros((args.rows, args.cols))
    s_r = CSVec(args.p, args.cols, args.rows, doInitialize=False, device='cpu', numBlocks=1)
    s_r.buckets, s_r.signs = S_e.buckets, S_e.signs
    s_r.table = torch.zeros((args.rows, args.cols))
    S = CSVec(args.p, args.cols, args.rows, doInitialize=False, device='cpu', numBlocks=1)
    S.buckets, S.signs = S_e.buckets, S_e.signs
    S.table = torch.zeros((args.rows, args.cols))
    #initialize
    csvecs = [CSVec(args.p, args.cols, args.rows, doInitialize=False, device='cpu', numBlocks=1) for _ in range(args.user_number)]
    for i, csvec in enumerate(csvecs):
        csvec.buckets, csvec.signs = S_e.buckets, S_e.signs
        csvec.table = torch.zeros((args.rows, args.cols))
    for epoch in range(args.num_epochs):
        y = np.zeros((args.rows,  args.cols))
        agg_grad = np.zeros(args.p)
        for i, network in enumerate(networks):
            # Compute local gradient
            user_grad, user_final_shape, user_initial_shape = networks[i].grad_compute_loader_reg(data_loaders[i], networks[i].DNN_model, reference_params)
            user_grad = np.multiply(args.learning_rate, user_grad)
            args.local_grad_fsgd[i, epoch, : ] = user_grad
            agg_grad  += user_grad
            #user_grad = attenuate_array(user_grad, args.k, args.attenuation_factor)
            csvecs[i].table = torch.zeros((args.rows, args.cols))
            csvecs[i].accumulateVec(torch.tensor(user_grad))
            y += csvecs[i].table.numpy()
        noise = np.random.normal(args.noise_mean, args.noise_std, (args.rows, args.cols))
        y += noise
        y = y/args.user_number
        args.agg_grad_fsgd[epoch, :] = (agg_grad + np.random.normal(args.noise_mean, args.noise_std, args.p))/args.user_number
        s_r.table = torch.zeros((args.rows, args.cols))
        s_r.table = torch.tensor(y)
        S_e.table = S_e.table + s_r.table
        unSketched_rx = S_e.unSketch(k=args.k)
        S.table = torch.zeros((args.rows, args.cols))
        S.accumulateVec(unSketched_rx)
        nz = S.table.nonzero()
        S_e.table[nz[:, 0], nz[:, 1]] = 0
        unsketched_rx_np = unSketched_rx.numpy()
        #user_grad = unsketched_rx_np
        for i, network in enumerate(networks):    
            networks[i].update_params(unsketched_rx_np)    
        batch_test, labels_test = get_next_batch(test_loader)
        accuracy = networks[0].check_accuracy_torch(batch_test, labels_test, epoch)
        if epoch % args.global_agg == 0 :
            fetchSGD_accuracy.append(accuracy)
            print(f'FetchSGD accuracy for average round = {args.current_avg_round}, regularization factor = {args.lambda_reg}, k = {args.k}, columns = {args.cols} and attenuation = {args.attenuation_factor}, at iteration:', epoch,'is: ', accuracy)
    return fetchSGD_accuracy




 
