import argparse
import os
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Pytorch implementation.")
    parser.add_argument('--cuda',  type=str, default='False', help='Availability of cuda')
    parser.add_argument('--user_number', type=int, default=50, help='Number of users in the network')  # 10
    parser.add_argument('--subchannel_number', type=int, default=50000, help='Number of channels per user')  # 1024
    parser.add_argument('--num_clients', type=int, default=20, help='Number of clients selected')  # 1024
    parser.add_argument('--train_batch_size', type=int, default=64)  # Must be less than number of samples in testing
    parser.add_argument('--test_batch_size', type=int, default=256)  # Must be less than number of samples in testing
    parser.add_argument('--iteration_number', type=int, default=2000)
    parser.add_argument('--beta_1', type=float, default=0.5)
    parser.add_argument('--beta_2', type=float, default=0.999)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--noise_dim_0', type=int, default=100)
    parser.add_argument('--data_folder', type=str, default='MNIST_data')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--train_classes', type=list, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--noise_mean', type=float, default=0)
    parser.add_argument('--noise_std', type=float, default=0)
    parser.add_argument('--save_folder', type=str, default='save_batch/data_folder')
    parser.add_argument('--mask_style', type=str, default='uniform')
    parser.add_argument('--eps', type=float, default=0.000001)
    parser.add_argument('--rows', type=int, default=3)
    parser.add_argument('--k', type=int, default=20000)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--dataset', type=str, default='kdd10')
    parser.add_argument('--dist', type=str, default='non_iid')#'non_iid_extreme' #non_iid, iid, non_iid_extreme
    parser.add_argument('--avg', type=int, default=3)
    parser.add_argument('--local_epoch', type=int, default=5)
    parser.add_argument('--global_agg', type=int, default=5)
    parser.add_argument('--timestamp', type=str, default=str(datetime.now()))
    parser.add_argument('--DNN_style', type=str, default='simple')
    
    # distributed, error_free, centralized, single_user, equal_power, normal_error
    if not os.path.exists('save_batch'):
        os.mkdir('save_batch')

    args = parser.parse_args()

    setattr(args, 'noise_dim', (args.noise_dim_0, 1, 1))

    return args


def save_module(arg, save_fld, iters, save_vector, vector_content):
    for i in range(len(vector_content)):
        if vector_content[i] == "gamma":
            file = save_fld + '/gamma.txt'
            gamma_save = save_vector[i]
            if not os.path.exists(file):
                f = open(file, "w+")
                f.write(str(gamma_save))
                f.close()
            else:
                f = open(file, "a")
                f.write(','+str(gamma_save))
                f.close()
        elif vector_content[i] == "accuracy":
            file = save_fld + '/accuracy.txt'
            acc_save = save_vector[i]
            if not os.path.exists(file):
                f = open(file, "w+")
                f.write(str(acc_save))
                f.close()
            else:
                f = open(file, "a")
                f.write(','+str(acc_save))
                f.close()

    file = save_fld + '/iterations.txt'
    if not os.path.exists(file):
        f = open(file, "w+")
        f.write(str(iters))
        f.close()
    else:
        f = open(file, "a")
        f.write(',' + str(iters))
        f.close()


def parse_args_fid():
    parser = argparse.ArgumentParser(description="Pytorch implementation.")

    args = parser.parse_args()

    file = open("figure_setup.txt", "r")
    labels = []
    legend_text = []
    iteration_files = []
    fid_files = []
    k = 0
    file = file.read().splitlines()
    for line in file:
        fields = line.split(";")
        if k == 0:
            labels.append(fields[0])
            labels.append(fields[1])
            labels.append(fields[2])
        elif k == 1:
            legend_text = fields
        elif k == 2:
            iteration_files = fields
        elif k == 3:
            fid_files = fields
        k += 1
    setattr(args, 'labels', labels)
    setattr(args, 'legend_text', legend_text)
    setattr(args, 'iteration_files', iteration_files)
    setattr(args, 'fid_files', fid_files)

    return args
