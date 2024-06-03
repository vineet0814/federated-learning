import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from DNN import DNN_v1, DNN_v2, DNN_v3, DNN_Resnet, ResNet9, SimpleNN
import numpy as np
torch.autograd.set_detect_anomaly(True)




class learning_model_reg(object):
    def __init__(self, args):
        #   print("Creating a new learning model...")
        if args.DNN_style == 'resnet':
            self.DNN_model = DNN_Resnet()
        elif args.DNN_style == 'v1':
            self.DNN_model = DNN_v1()
        elif args.DNN_style == 'v2':
            self.DNN_model = DNN_v2()
        elif args.DNN_style == 'resnet9':
            self.DNN_model = ResNet9()
        elif args.DNN_style == 'simple':
            self.DNN_model = SimpleNN()
        else:
            self.DNN_model = DNN_v3()
        for p in self.DNN_model.parameters():
            p.requires_grad = True
        #   print("Number of trainable parameters: ", sum(p.numel() for p in self.DNN_model.parameters() if p.requires_grad))
        self.optim = optim.Adam(self.DNN_model.parameters(), lr=args.learning_rate,
                                betas=(args.beta_1, args.beta_2), weight_decay = args.weight_decay)
        self.cuda_index = 0
        self.cuda = False
        self.check_cuda(args.cuda)
        self.batch_size = args.train_batch_size
        self.workers = args.workers
        self.loss_fn = self.CustomLoss(args.lambda_reg)
        self.grad_vector_final_shape = []
        self.grad_vector_initial_shape = []

    class CustomLoss(nn.Module):
        def __init__(self, lambda_reg):
            self.lambda_reg = lambda_reg

        def forward(self, output, target, model, reference_params):
            # Compute the cross-entropy loss
            ce_loss = nn.CrossEntropyLoss()(output, target)

            # Compute the regularization term as the sum of squared parameter differences
            reg_term = 0.0
            for param, ref_param in zip(model.parameters(), reference_params):
                param_diff = param - ref_param
                reg_term += torch.sum(param_diff ** 2)

            # Combine the cross-entropy loss and the regularization term
            total_loss = ce_loss + (self.lambda_reg * reg_term)

            return total_loss    

    def get_torch_variable(self, inp):
        if self.cuda:
            return Variable(inp).cuda(self.cuda_index)
        else:
            return Variable(inp)

    def check_cuda(self, cuda_flag=False):
        #   print("Cuda flag:", cuda_flag)
        if cuda_flag == 'True':
            self.cuda_index = 0
            self.cuda = True
            self.DNN_model.cuda(self.cuda_index)
            print("Models are assigned to GPU : {}".format(self.cuda_index))
        else:
            self.cuda = False

    def grad_compute(self, samples, labels):

        self.DNN_model.train()
        self.DNN_model.zero_grad()
        torch_samples = self.get_torch_variable(samples)
        torch_labels = self.get_torch_variable(labels)
        torch_output = self.DNN_model(torch_samples)
        torch_cost = self.loss_fn(torch_output, torch_labels)
        torch_cost.backward()
        # grad_values = self.DNN_model.parameters()

        # Convert gradient matrices to a single column vector
        params = list(self.DNN_model.parameters())
        grad_vector = []
        self.grad_vector_final_shape = []
        self.grad_vector_initial_shape = []
        for p_ind in range(len(params)):
            temp_vector = params[p_ind].grad.cpu().numpy()
            initial_shape = temp_vector.shape
            new_shape = 1
            for j in range(len(initial_shape)):
                new_shape = new_shape*initial_shape[j]
            final_vector = temp_vector.reshape(new_shape)
            grad_vector.extend(final_vector)
            self.grad_vector_final_shape.append(new_shape)
            self.grad_vector_initial_shape.append(initial_shape)

        out_vector = np.asarray(grad_vector)

        return out_vector, self.grad_vector_final_shape, self.grad_vector_initial_shape

    def grad_compute_loader_reg(self, loader, model, reference_params):

        self.DNN_model.train()
        self.DNN_model.zero_grad()
        torch_samples, torch_labels = self.get_next_batch(loader)
        #torch_samples = self.get_torch_variable(samples)
        #torch_labels = self.get_torch_variable(labels)
        torch_output = self.DNN_model(torch_samples)
        torch_cost = self.loss_fn.forward(torch_output, torch_labels, model, reference_params)
        torch_cost.backward()
        # grad_values = self.DNN_model.parameters()

        # Convert gradient matrices to a single column vector
        params = list(self.DNN_model.parameters())
        grad_vector = []
        self.grad_vector_final_shape = []
        self.grad_vector_initial_shape = []
        for p_ind in range(len(params)):
            temp_vector = params[p_ind].grad.cpu().numpy()
            initial_shape = temp_vector.shape
            new_shape = 1
            for j in range(len(initial_shape)):
                new_shape = new_shape*initial_shape[j]
            final_vector = temp_vector.reshape(new_shape)
            grad_vector.extend(final_vector)
            self.grad_vector_final_shape.append(new_shape)
            self.grad_vector_initial_shape.append(initial_shape)

        out_vector = np.asarray(grad_vector)

        return out_vector, self.grad_vector_final_shape, self.grad_vector_initial_shape
    def get_next_batch(self, loader):
        try:
            data_iter = iter(loader)
            inputs, labels = next(data_iter)
            return inputs, labels
        except StopIteration:
            # If there are no more batches, create a new iterator to wrap around
            data_iter = iter(loader)
            inputs, labels = next(data_iter)
            return inputs, labels

    def update_params(self, grad_param):
        self.DNN_model.train()
        grad_param_torch = self.get_torch_variable(torch.Tensor(grad_param))
        initial_index = 0
        params = list(self.DNN_model.parameters())
        with torch.no_grad():
            for ell in range(len(self.grad_vector_final_shape)):
                # p.grad = grad[i].clone()
                grads = grad_param_torch[initial_index:initial_index + self.grad_vector_final_shape[ell]]\
                    .reshape(self.grad_vector_initial_shape[ell]).clone()
                # params[ell].grad = grad_param_torch[initial_index:initial_index + self.grad_vector_final_shape[ell]]\
                #     .reshape(self.grad_vector_initial_shape[ell]).clone()
                initial_index = initial_index + self.grad_vector_final_shape[ell]
                new_val = params[ell] - grads
                params[ell].copy_(new_val)

        #   print('params updated')
        # with torch.no_grad():
        #     for p in model.parameters():
        #         new_val = update_function(p, p.grad, loss, other_params)
        #         p.copy_(new_val)
        # # for p in self.DNN_model.parameters():
        # #     p.grad *=
        # self.optim.step()
    def update_weights(self, weight_param):
        self.DNN_model.train()
        weight_param_torch = self.get_torch_variable(torch.Tensor(weight_param))
        initial_index = 0
        params = list(self.DNN_model.parameters())
        with torch.no_grad():
            for ell in range(len(self.grad_vector_final_shape)):
                # p.grad = grad[i].clone()
                weights = weight_param_torch[initial_index:initial_index + self.grad_vector_final_shape[ell]]\
                    .reshape(self.grad_vector_initial_shape[ell]).clone()
                # params[ell].grad = grad_param_torch[initial_index:initial_index + self.grad_vector_final_shape[ell]]\
                #     .reshape(self.grad_vector_initial_shape[ell]).clone()
                initial_index = initial_index + self.grad_vector_final_shape[ell]
                new_val = weights
                params[ell].copy_(new_val)

        #   print('params updated')
        # with torch.no_grad():
        #     for p in model.parameters():
        #         new_val = update_function(p, p.grad, loss, other_params)
        #         p.copy_(new_val)
        # # for p in self.DNN_model.parameters():
        # #     p.grad *=
        # self.optim.step()
    def check_accuracy(self, batch_test, labels_test, iteration):
        self.DNN_model.eval()
        correct = 0
        total = 0
        torch_samples = self.get_torch_variable(batch_test)
        torch_labels = self.get_torch_variable(labels_test)
        torch_output = self.DNN_model(torch_samples)

        _, predicted = torch.max(torch_output.data, 1)
        total += torch_labels.size(0)
        correct += (predicted == torch_labels).sum().item()
        accuracy = correct/total
        print('Accuracy at iteration ', str(iteration), ' is: ', str(accuracy))
        return accuracy
    def check_accuracy_torch(self, torch_samples, torch_labels, iteration):
        self.DNN_model.eval()
        correct = 0
        total = 0
        #torch_samples = self.get_torch_variable(batch_test)
        #torch_labels = self.get_torch_variable(labels_test)
        torch_output = self.DNN_model(torch_samples)

        _, predicted = torch.max(torch_output.data, 1)
        total += torch_labels.size(0)
        correct += (predicted == torch_labels).sum().item()
        accuracy = correct/total
        #print('Accuracy at iteration ', str(iteration), ' is: ', str(accuracy))
        return accuracy

    
    def multi_acc(self, batch_test, labels_test):
        self.DNN_model.eval()
        correct = 0
        total = 0
        torch_samples = self.get_torch_variable(batch_test)
        torch_labels = self.get_torch_variable(labels_test)
        torch_output = self.DNN_model(torch_samples)
        y_pred_softmax = torch.log_softmax(torch_output, dim = 1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
        correct_pred = (y_pred_tags == labels_test).float()
        acc = correct_pred.sum() / len(correct_pred)
    
        acc = torch.round(acc * 100)
    
        return acc










