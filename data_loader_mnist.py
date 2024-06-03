import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, Subset
import os
import numpy as np


def get_data_loader(args):
    # Define a transform to preprocess the data (you can customize this further)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    # Download and load the MNIST dataset
    mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    # Number of subsets 
    num_subsets = args.user_number
    
    if args.dist == 'iid':
        
        # Split the MNIST dataset into 10 IID subsets
        
        subset_size = len(mnist_dataset) // num_subsets
        subset_indices = [list(range(i * subset_size, (i + 1) * subset_size)) for i in range(num_subsets)]
    
        # Create DataLoaders for each subset
        data_loaders = []
    
        for indices in subset_indices:
            subset = Subset(mnist_dataset, indices)
            data_loader = DataLoader(subset, batch_size=args.train_batch_size, shuffle=True)
            data_loaders.append(data_loader)
    
    elif args.dist == 'non_iid':
    
    
        # Define the Dirichlet distribution hyperparameter (alpha)
        alpha = args.alpha
    
        # Initialize subsets
        subsets = [Subset(mnist_dataset, []) for _ in range(num_subsets)]
    
        # Calculate proportions of samples for each class
        num_classes = 10
        proportions = np.random.dirichlet([alpha] * num_subsets, num_classes)
    
        # Assign samples to subsets based on the proportions
        for class_idx in range(num_classes):
            class_indices = np.where(np.array(mnist_dataset.targets) == class_idx)[0]
            num_samples = len(class_indices)
            class_proportions = proportions[class_idx] / proportions[class_idx].sum()  # Normalize proportions for this class
    
            # Shuffle the class indices randomly
            np.random.shuffle(class_indices)
    
            # Assign samples to subsets
            start = 0
            for i in range(num_subsets):
                num_samples_to_assign = int(class_proportions[i] * num_samples)
                end = start + num_samples_to_assign
                subsets[i].indices.extend(class_indices[start:end])
                start = end
    
        # Create DataLoaders for each subset
        data_loaders = [DataLoader(subset, batch_size=args.train_batch_size, shuffle=True) for subset in subsets]
    
    
    
    elif args.dist == 'non_iid_extreme':
    
        # Define the Dirichlet distribution hyperparameter (alpha)
        alpha = args.alpha
    
        # Initialize subsets
        subsets = [Subset(mnist_dataset, []) for _ in range(num_subsets)]
    
        # Calculate proportions of samples for each class
        num_classes = 10
        proportions = np.eye(num_classes) * 1
    
        # Set the elements above the diagonal by one index to 0.5
        #for j in range(num_classes):
        #    proportions[j, (j + 1) % 10] = 0.5
    
        # Assign samples to subsets based on the proportions
        for class_idx in range(num_classes):
            class_indices = np.where(np.array(mnist_dataset.targets) == class_idx)[0]
            num_samples = len(class_indices)
            class_proportions = proportions[class_idx] / proportions[class_idx].sum()  # Normalize proportions for this class
    
            # Shuffle the class indices randomly
            np.random.shuffle(class_indices)
    
            # Assign samples to subsets
            start = 0
            for i in range(num_subsets):
                num_samples_to_assign = int(class_proportions[i] * num_samples)
                end = start + num_samples_to_assign
                subsets[i].indices.extend(class_indices[start:end])
                start = end
    
        # Create DataLoaders for each subset
        data_loaders = [DataLoader(subset, batch_size=args.train_batch_size, shuffle=True) for subset in subsets]
        
    # Load the MNIST test dataset
    test_data = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False)

    return data_loaders, test_loader




def get_next_batch(loader):
    try:
        data_iter = iter(loader)
        inputs, labels = next(data_iter)
        return inputs, labels
    except StopIteration:
        # If there are no more batches, create a new iterator to wrap around
        data_iter = iter(loader)
        inputs, labels = next(data_iter)
        return inputs, labels
