import os
import numpy as np
import torch.nn as nn
from main_new_OG import *
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
import random
from sklearn.decomposition import IncrementalPCA
from main import *
import h5py
import xarray as xr
from torch_cv import *
from regression import *
from sklearn.linear_model import Ridge
import shutil
from scipy.stats import zscore
from tqdm import tqdm
from helpers.encoding_score_tools_new import *
from sklearn.decomposition import NMF  # Add this import at the top of your file
from sklearn.decomposition import FastICA
import os
import random
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
import numpy as np
from sklearn.decomposition import IncrementalPCA

def generate_model_to_compute_mean_and_var(num_channels, var_path, std_path, pca_model=True):
        
    model = load_model_modified(num_channels)
    n = len([f for f in os.listdir(var_path) if f.startswith('bias')])
    
    if n > 0: 
        new_num_channels = num_channels[:n]
        if pca_model:
            m1 = make_pca(new_num_channels, var_path, std_path)
        else:
            m1 = make_random(new_num_channels, std_path)
        
        new_model = nn.Sequential(
            model[0], 
            *[m1[i] for i in range(1, len(new_num_channels) + 1)], 
            model[len(new_num_channels) + 1][:2]
        )
    else: 
        new_model = nn.Sequential(
            model[0], 
            model[1][:2]
        )
    
    return new_model

def generate_model_to_compute_pm_and_bias(num_channels, var_path, std_path, pca_model=True):
    n = len([f for f in os.listdir(std_path) if f.startswith('mean')])
    if n % 2 == 1:
        model = load_model_new(num_channels[:n], std_path)        
        if n > 1:
            if pca_model:
                m1 = make_pca(num_channels[:n-1], var_path, std_path)
            else:
                m1 = make_random(num_channels[:n-1], std_path)
            new_model = nn.Sequential(
                model[0],  # First layer from model
                *[m1[i] for i in range(1, len(num_channels[:n-1]) + 1)], 
                model[len(num_channels[:n-1]) + 1][:3]  # Final layers from model
            )
        else:
            new_model = nn.Sequential(
                model[0],  # First layer from model
                model[1][:3]  # Final layers from model
            )
    else:
        dummy_mean_path = os.path.join(std_path, f'mean{n+1}.npy')
        dummy_var_path = os.path.join(std_path, f'var{n+1}.npy')
        np.save(dummy_mean_path, np.random.randn(1))  # Create dummy mean file
        np.save(dummy_var_path, np.random.randn(1))  # Create dummy variance file
        model = load_model_new(num_channels[:n+1], std_path)
        if pca_model:
            m1 = make_pca(num_channels[:n-1], var_path, std_path)
        else:
            m1 = make_random(num_channels[:n-1], std_path)
        new_model = nn.Sequential(
            model[0],  # First layer from model
            *[m1[i] for i in range(1, len(num_channels[:n-1]) + 1)],  # Layers from m1
            model[len(num_channels[:n-1]) + 1][:3]  # Final layers from model
        )
        os.remove(dummy_mean_path)
        os.remove(dummy_var_path)      
    return new_model

def train_model(channel_sizes, num_training_images, var_path, std_path, pca_model=True, seed=42):

    if not os.path.exists(var_path):
        os.makedirs(var_path)
        
    if not os.path.exists(std_path):
        os.makedirs(std_path)    

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    random.seed(seed)
    torch.manual_seed(seed)
    folder_path = "/data/shared/datasets/imagenet"
    dataset = ImageFolder(root=folder_path, transform=transform)
    dataset_length = len(dataset)
    subset_size = num_training_images
    bs = 30
    subset_indices = random.sample(range(dataset_length), subset_size)
    dataset = Subset(dataset, subset_indices)
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True)

    for layerid in range(1, len(channel_sizes) + 1):
        # Pass the pca_model flag to use the proper model generation function
        model = generate_model_to_compute_mean_and_var(channel_sizes, var_path, std_path, pca_model=pca_model)        
        model.cuda()
        m = []
        j = 1
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs
            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            m.append(output)

        m = torch.cat(m, dim=0)
        m = torch.mean(m, dim=0)
        m = m.view(1, len(m), 1, 1)
        m = m.cuda()

        std = []
        j = 0
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs
            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = output - m
            output = output ** 2
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            std.append(output)

        std = torch.cat(std, dim=0)
        std = torch.mean(std, dim=0)
        m = m.cpu().squeeze()
        std = std.cpu()

        np.save(os.path.join(std_path, f'mean{layerid}.npy'), m.numpy())
        np.save(os.path.join(std_path, f'var{layerid}.npy'), std.numpy())

        # Generate model for projection matrix and bias computation
        model = generate_model_to_compute_pm_and_bias(channel_sizes, var_path, std_path, pca_model=pca_model)
        model.cuda()
        coefs = []
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs

            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            coefs.append(output)

        coefs = torch.cat(coefs, dim=0)

        # Mean centering
        channel_mean = torch.mean(coefs, axis=0)
        coefs = coefs - channel_mean

        if pca_model:
            # Compute PCA on the coefficients
            pca = IncrementalPCA()
            pca.fit(coefs)
            # Get the projection matrix and bias from PCA
            projection_matrix = pca.components_
            projection_matrix = torch.tensor(projection_matrix, dtype=torch.float32)
            bias = projection_matrix @ channel_mean
            bias = -1 * bias
        else:
            # For random models, generate a random projection matrix and use zero bias (example)
            num_features = coefs.shape[1]
            projection_matrix = torch.randn(num_features, num_features, dtype=torch.float32)
            bias = torch.zeros(num_features, dtype=torch.float32)

        np.save(os.path.join(var_path, f'pm_cent{layerid}.npy'), projection_matrix.numpy())
        np.save(os.path.join(var_path, f'bias{layerid}.npy'), bias.numpy())

        print(f"Layer {layerid} completed.")
        
from scipy.stats import ortho_group  # Add this import at the top

def train_orthogonal_model(channel_sizes, num_training_images, var_path, std_path, seed=42):

    if not os.path.exists(var_path):
        os.makedirs(var_path)
        
    if not os.path.exists(std_path):
        os.makedirs(std_path)    

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    random.seed(seed)
    torch.manual_seed(seed)
    folder_path = "/data/shared/datasets/imagenet"
    dataset = ImageFolder(root=folder_path, transform=transform)
    dataset_length = len(dataset)
    subset_size = num_training_images
    bs = 30
    subset_indices = random.sample(range(dataset_length), subset_size)
    dataset = Subset(dataset, subset_indices)
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True)

    for layerid in range(1, len(channel_sizes) + 1):
        # Pass the pca_model flag to use the proper model generation function
        model = generate_model_to_compute_mean_and_var(channel_sizes, var_path, std_path, pca_model=True)        
        model.cuda()
        m = []
        j = 1
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs
            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            m.append(output)

        m = torch.cat(m, dim=0)
        m = torch.mean(m, dim=0)
        m = m.view(1, len(m), 1, 1)
        m = m.cuda()

        std = []
        j = 0
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs
            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = output - m
            output = output ** 2
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            std.append(output)

        std = torch.cat(std, dim=0)
        std = torch.mean(std, dim=0)
        m = m.cpu().squeeze()
        std = std.cpu()

        np.save(os.path.join(std_path, f'mean{layerid}.npy'), m.numpy())
        np.save(os.path.join(std_path, f'var{layerid}.npy'), std.numpy())

        # --- Orthogonal pm and bias calculation ---
        model = generate_model_to_compute_pm_and_bias(channel_sizes, var_path, std_path, pca_model=True)
        model.cuda()
        coefs = []
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs

            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            coefs.append(output)

        coefs = torch.cat(coefs, dim=0)

        # Mean centering
        channel_mean = torch.mean(coefs, axis=0)  # shape: [num_features]
        coefs = coefs - channel_mean

        num_features = coefs.shape[1]
        # Generate random orthogonal matrix
        projection_matrix = ortho_group.rvs(dim=num_features).astype(np.float32)
        projection_matrix_torch = torch.from_numpy(projection_matrix)  # shape: [num_features, num_features]

        # Calculate bias as in original PCA path
        bias = -1 * torch.matmul(projection_matrix_torch, channel_mean)

        np.save(os.path.join(var_path, f'pm_cent{layerid}.npy'), projection_matrix)
        np.save(os.path.join(var_path, f'bias{layerid}.npy'), bias.numpy())

        print(f"Layer {layerid} completed.")
        
def train_model_flexible(channel_sizes, num_training_images, var_path, std_path, folder_path, pca_model=True, seed=42):

    if not os.path.exists(var_path):
        os.makedirs(var_path)
        
    if not os.path.exists(std_path):
        os.makedirs(std_path)    

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    random.seed(seed)
    torch.manual_seed(seed)
    dataset = ImageFolder(root=folder_path, transform=transform)
    dataset_length = len(dataset)
    subset_size = num_training_images
    bs = 30
    subset_indices = random.sample(range(dataset_length), subset_size)
    dataset = Subset(dataset, subset_indices)
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True)

    for layerid in range(1, len(channel_sizes) + 1):
        # Pass the pca_model flag to use the proper model generation function
        model = generate_model_to_compute_mean_and_var(channel_sizes, var_path, std_path, pca_model=pca_model)        
        model.cuda()
        m = []
        j = 1
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs
            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            m.append(output)

        m = torch.cat(m, dim=0)
        m = torch.mean(m, dim=0)
        m = m.view(1, len(m), 1, 1)
        m = m.cuda()

        std = []
        j = 0
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs
            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = output - m
            output = output ** 2
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            std.append(output)

        std = torch.cat(std, dim=0)
        std = torch.mean(std, dim=0)
        m = m.cpu().squeeze()
        std = std.cpu()

        np.save(os.path.join(std_path, f'mean{layerid}.npy'), m.numpy())
        np.save(os.path.join(std_path, f'var{layerid}.npy'), std.numpy())

        # Generate model for projection matrix and bias computation
        if pca_model:
            model = generate_model_to_compute_pm_and_bias(channel_sizes, var_path, std_path, pca_model=pca_model)
            model.cuda()
            coefs = []
            for batch_input, _ in data_loader:
                batch_input = Variable(batch_input.cuda())
                print(round((j / subset_size) * 100, 2), end="\r")
                j += bs

                with torch.no_grad():
                    output = model(batch_input)
                output = output.full_view()
                output = torch.mean(output, axis=(2, 3))
                output = output.cpu()
                coefs.append(output)

            coefs = torch.cat(coefs, dim=0)

            # Mean centering
            channel_mean = torch.mean(coefs, axis=0)
            coefs = coefs - channel_mean

            # Compute PCA on the coefficients
            pca = IncrementalPCA()
            pca.fit(coefs)
            # Get the projection matrix and bias from PCA
            projection_matrix = pca.components_
            projection_matrix = torch.tensor(projection_matrix, dtype=torch.float32)
            bias = projection_matrix @ channel_mean
            bias = -1 * bias
        else:
            num_features = 5
            # For random models, generate a random projection matrix and use zero bias (example)
            projection_matrix = torch.randn(num_features, num_features, dtype=torch.float32)
            bias = torch.zeros(num_features, dtype=torch.float32)

        np.save(os.path.join(var_path, f'pm_cent{layerid}.npy'), projection_matrix.numpy())
        np.save(os.path.join(var_path, f'bias{layerid}.npy'), bias.numpy())

        print(f"Layer {layerid} completed.")
        
def train_model_5k(channel_sizes, num_training_images, var_path, std_path, pca_model=True, seed=42):

    if not os.path.exists(var_path):
        os.makedirs(var_path)
        
    if not os.path.exists(std_path):
        os.makedirs(std_path)    

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    random.seed(seed)
    torch.manual_seed(seed)
    folder_path = "/data/shared/datasets/imagenet-5k"
    dataset = ImageFolder(root=folder_path, transform=transform)
    dataset_length = len(dataset)
    subset_size = num_training_images
    bs = 30
    subset_indices = random.sample(range(dataset_length), subset_size)
    dataset = Subset(dataset, subset_indices)
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True)

    for layerid in range(1, len(channel_sizes) + 1):
        # Pass the pca_model flag to use the proper model generation function
        model = generate_model_to_compute_mean_and_var(channel_sizes, var_path, std_path, pca_model=pca_model)        
        model.cuda()
        m = []
        j = 1
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs
            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            m.append(output)

        m = torch.cat(m, dim=0)
        m = torch.mean(m, dim=0)
        m = m.view(1, len(m), 1, 1)
        m = m.cuda()

        std = []
        j = 0
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs
            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = output - m
            output = output ** 2
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            std.append(output)

        std = torch.cat(std, dim=0)
        std = torch.mean(std, dim=0)
        m = m.cpu().squeeze()
        std = std.cpu()

        np.save(os.path.join(std_path, f'mean{layerid}.npy'), m.numpy())
        np.save(os.path.join(std_path, f'var{layerid}.npy'), std.numpy())

        # Generate model for projection matrix and bias computation
        model = generate_model_to_compute_pm_and_bias(channel_sizes, var_path, std_path, pca_model=pca_model)
        model.cuda()
        coefs = []
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs

            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            coefs.append(output)

        coefs = torch.cat(coefs, dim=0)

        # Mean centering
        channel_mean = torch.mean(coefs, axis=0)
        coefs = coefs - channel_mean

        if pca_model:
            # Compute PCA on the coefficients
            pca = IncrementalPCA()
            pca.fit(coefs)
            # Get the projection matrix and bias from PCA
            projection_matrix = pca.components_
            projection_matrix = torch.tensor(projection_matrix, dtype=torch.float32)
            bias = projection_matrix @ channel_mean
            bias = -1 * bias
        else:
            # For random models, generate a random projection matrix and use zero bias (example)
            num_features = coefs.shape[1]
            projection_matrix = torch.randn(num_features, num_features, dtype=torch.float32)
            bias = torch.zeros(num_features, dtype=torch.float32)

        np.save(os.path.join(var_path, f'pm_cent{layerid}.npy'), projection_matrix.numpy())
        np.save(os.path.join(var_path, f'bias{layerid}.npy'), bias.numpy())

        print(f"Layer {layerid} completed.")
        
def train_model_THINGS(channel_sizes, num_training_images, var_path, std_path, pca_model=True, seed=42):

    if not os.path.exists(var_path):
        os.makedirs(var_path)
        
    if not os.path.exists(std_path):
        os.makedirs(std_path)    

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    random.seed(seed)
    torch.manual_seed(seed)
    folder_path = "/data/shared/datasets/hebart2019.things/images/object_images"
    dataset = ImageFolder(root=folder_path, transform=transform)
    dataset_length = len(dataset)
    subset_size = num_training_images
    bs = 30
    subset_indices = random.sample(range(dataset_length), subset_size)
    dataset = Subset(dataset, subset_indices)
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True)

    for layerid in range(1, len(channel_sizes) + 1):
        # Pass the pca_model flag to use the proper model generation function
        model = generate_model_to_compute_mean_and_var(channel_sizes, var_path, std_path, pca_model=pca_model)        
        model.cuda()
        m = []
        j = 1
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs
            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            m.append(output)

        m = torch.cat(m, dim=0)
        m = torch.mean(m, dim=0)
        m = m.view(1, len(m), 1, 1)
        m = m.cuda()

        std = []
        j = 0
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs
            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = output - m
            output = output ** 2
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            std.append(output)

        std = torch.cat(std, dim=0)
        std = torch.mean(std, dim=0)
        m = m.cpu().squeeze()
        std = std.cpu()

        np.save(os.path.join(std_path, f'mean{layerid}.npy'), m.numpy())
        np.save(os.path.join(std_path, f'var{layerid}.npy'), std.numpy())

        # Generate model for projection matrix and bias computation
        model = generate_model_to_compute_pm_and_bias(channel_sizes, var_path, std_path, pca_model=pca_model)
        model.cuda()
        coefs = []
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs

            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            coefs.append(output)

        coefs = torch.cat(coefs, dim=0)

        # Mean centering
        channel_mean = torch.mean(coefs, axis=0)
        coefs = coefs - channel_mean

        if pca_model:
            # Compute PCA on the coefficients
            pca = IncrementalPCA()
            pca.fit(coefs)
            # Get the projection matrix and bias from PCA
            projection_matrix = pca.components_
            projection_matrix = torch.tensor(projection_matrix, dtype=torch.float32)
            bias = projection_matrix @ channel_mean
            bias = -1 * bias
        else:
            # For random models, generate a random projection matrix and use zero bias (example)
            num_features = coefs.shape[1]
            projection_matrix = torch.randn(num_features, num_features, dtype=torch.float32)
            bias = torch.zeros(num_features, dtype=torch.float32)

        np.save(os.path.join(var_path, f'pm_cent{layerid}.npy'), projection_matrix.numpy())
        np.save(os.path.join(var_path, f'bias{layerid}.npy'), bias.numpy())

        print(f"Layer {layerid} completed.")
        
def train_model_new(channel_sizes, num_training_images, var_path, std_path, folder_path, pca_model=True, seed=42):

    if not os.path.exists(var_path):
        os.makedirs(var_path)
        
    if not os.path.exists(std_path):
        os.makedirs(std_path)    

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    random.seed(seed)
    torch.manual_seed(seed)

    # Use the user-supplied path for images.
    dataset = ImageFolder(root=folder_path, transform=transform)
    dataset_length = len(dataset)
    subset_size = num_training_images
    bs = 30
    subset_indices = random.sample(range(dataset_length), subset_size)
    dataset = Subset(dataset, subset_indices)
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True)

    for layerid in range(1, len(channel_sizes) + 1):
        model = generate_model_to_compute_mean_and_var(channel_sizes, var_path, std_path, pca_model=pca_model)        
        model.cuda()
        m = []
        j = 1
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs
            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            m.append(output)

        m = torch.cat(m, dim=0)
        m = torch.mean(m, dim=0)
        m = m.view(1, len(m), 1, 1)
        m = m.cuda()

        std = []
        j = 0
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs
            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = output - m
            output = output ** 2
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            std.append(output)

        std = torch.cat(std, dim=0)
        std = torch.mean(std, dim=0)
        m = m.cpu().squeeze()
        std = std.cpu()

        np.save(os.path.join(std_path, f'mean{layerid}.npy'), m.numpy())
        np.save(os.path.join(std_path, f'var{layerid}.npy'), std.numpy())

        model = generate_model_to_compute_pm_and_bias(channel_sizes, var_path, std_path, pca_model=pca_model)
        model.cuda()
        coefs = []
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs

            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            coefs.append(output)

        coefs = torch.cat(coefs, dim=0)

        # Mean centering
        channel_mean = torch.mean(coefs, axis=0)
        coefs = coefs - channel_mean

        if pca_model:
            pca = IncrementalPCA()
            pca.fit(coefs)
            projection_matrix = pca.components_
            projection_matrix = torch.tensor(projection_matrix, dtype=torch.float32)
            bias = projection_matrix @ channel_mean
            bias = -1 * bias
        else:
            num_features = coefs.shape[1]
            projection_matrix = torch.randn(num_features, num_features, dtype=torch.float32)
            bias = torch.zeros(num_features, dtype=torch.float32)

        np.save(os.path.join(var_path, f'pm_cent{layerid}.npy'), projection_matrix.numpy())
        np.save(os.path.join(var_path, f'bias{layerid}.npy'), bias.numpy())

        print(f"Layer {layerid} completed.")
        
def train_model_with_NMF(channel_sizes, num_training_images, var_path, std_path, pca_model=True, seed=42):

    if not os.path.exists(var_path):
        os.makedirs(var_path)
        
    if not os.path.exists(std_path):
        os.makedirs(std_path)    

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    random.seed(seed)
    torch.manual_seed(seed)
    folder_path = "/data/shared/datasets/imagenet"
    dataset = ImageFolder(root=folder_path, transform=transform)
    dataset_length = len(dataset)
    subset_size = num_training_images
    bs = 30
    subset_indices = random.sample(range(dataset_length), subset_size)
    dataset = Subset(dataset, subset_indices)
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True)

    for layerid in range(1, len(channel_sizes) + 1):
        # Pass the pca_model flag to use the proper model generation function
        model = generate_model_to_compute_mean_and_var(channel_sizes, var_path, std_path, pca_model=pca_model)        
        model.cuda()
        m = []
        j = 1
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs
            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            m.append(output)

        m = torch.cat(m, dim=0)
        m = torch.mean(m, dim=0)
        m = m.view(1, len(m), 1, 1)
        m = m.cuda()

        std = []
        j = 0
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs
            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = output - m
            output = output ** 2
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            std.append(output)

        std = torch.cat(std, dim=0)
        std = torch.mean(std, dim=0)
        m = m.cpu().squeeze()
        std = std.cpu()

        np.save(os.path.join(std_path, f'mean{layerid}.npy'), m.numpy())
        np.save(os.path.join(std_path, f'var{layerid}.npy'), std.numpy())

        # Generate model for projection matrix and bias computation
        model = generate_model_to_compute_pm_and_bias(channel_sizes, var_path, std_path, pca_model=pca_model)
        model.cuda()
        coefs = []
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs

            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            coefs.append(output)

        coefs = torch.cat(coefs, dim=0)

        # Mean centering removed for NMF. If you have negative values, clamp:
        coefs = coefs.clamp(min=0)  # Ensure non-negativity

        if pca_model:
            # === NMF in place of PCA ===
            num_features = coefs.shape[1]
            # Let number of components = num_features for parity; adjust if you want
            nmf = NMF(n_components=num_features, init='random', random_state=seed)
            W = nmf.fit_transform(coefs.numpy())
            projection_matrix = nmf.components_  # shape: (n_components, n_features)
            projection_matrix = torch.tensor(projection_matrix, dtype=torch.float32)
            bias = torch.zeros(projection_matrix.shape[0], dtype=torch.float32)
        else:
            # For random models, generate a random projection matrix and use zero bias (example)
            num_features = coefs.shape[1]
            projection_matrix = torch.randn(num_features, num_features, dtype=torch.float32)
            bias = torch.zeros(num_features, dtype=torch.float32)

        np.save(os.path.join(var_path, f'pm_cent{layerid}.npy'), projection_matrix.numpy())
        np.save(os.path.join(var_path, f'bias{layerid}.npy'), bias.numpy())

        print(f"Layer {layerid} completed.")
        
def zscore_train_model(channel_sizes, num_training_images, var_path, std_path, pca_model=True, seed=42):

    if not os.path.exists(var_path):
        os.makedirs(var_path)
        
    if not os.path.exists(std_path):
        os.makedirs(std_path)    

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    random.seed(seed)
    torch.manual_seed(seed)
    folder_path = "/data/shared/datasets/imagenet"
    dataset = ImageFolder(root=folder_path, transform=transform)
    dataset_length = len(dataset)
    subset_size = num_training_images
    bs = 30
    subset_indices = random.sample(range(dataset_length), subset_size)
    dataset = Subset(dataset, subset_indices)
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True)

    for layerid in range(1, len(channel_sizes) + 1):
        # Pass the pca_model flag to use the proper model generation function
        model = generate_model_to_compute_mean_and_var(channel_sizes, var_path, std_path, pca_model=pca_model)        
        model.cuda()
        m = []
        j = 1
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs
            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            m.append(output)

        m = torch.cat(m, dim=0)
        m = torch.mean(m, dim=0)
        m = m.view(1, len(m), 1, 1)
        m = m.cuda()

        std = []
        j = 0
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs
            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = output - m
            output = output ** 2
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            std.append(output)

        std = torch.cat(std, dim=0)
        std = torch.mean(std, dim=0)
        m = m.cpu().squeeze()
        std = std.cpu()

        np.save(os.path.join(std_path, f'mean{layerid}.npy'), m.numpy())
        np.save(os.path.join(std_path, f'var{layerid}.npy'), std.numpy())

        # Generate model for projection matrix and bias computation
        model = generate_model_to_compute_pm_and_bias(channel_sizes, var_path, std_path, pca_model=pca_model)
        model.cuda()
        coefs = []
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs

            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            coefs.append(output)

        coefs = torch.cat(coefs, dim=0)

        # Mean centering
        channel_mean = torch.mean(coefs, axis=0)
        coefs = coefs - channel_mean

        if pca_model:
            # Compute PCA on the coefficients
            coefs = zscore(coefs, axis=0, ddof=0)
            pca = IncrementalPCA()
            pca.fit(coefs)
            # Get the projection matrix and bias from PCA
            projection_matrix = pca.components_
            projection_matrix = torch.tensor(projection_matrix, dtype=torch.float32)
            bias = projection_matrix @ channel_mean
            bias = -1 * bias
        else:
            # For random models, generate a random projection matrix and use zero bias (example)
            num_features = coefs.shape[1]
            projection_matrix = torch.randn(num_features, num_features, dtype=torch.float32)
            bias = torch.zeros(num_features, dtype=torch.float32)

        np.save(os.path.join(var_path, f'pm_cent{layerid}.npy'), projection_matrix.numpy())
        np.save(os.path.join(var_path, f'bias{layerid}.npy'), bias.numpy())

        print(f"Layer {layerid} completed.")
        
def train_model_nsd(channel_sizes, var_path, std_path, pca_model=True):

    if not os.path.exists(var_path):
        os.makedirs(var_path)
    if not os.path.exists(std_path):
        os.makedirs(std_path)

    # Load all NSD images: unshared and shared, then combine
    data_array_unshared, data_array_shared = load_images(subjid=0)
    data_array_combined = torch.cat((data_array_unshared, data_array_shared), dim=0)
    subset_size = data_array_combined.shape[0]
    bs = 30  # Fixed batch size

    for layerid in range(1, len(channel_sizes) + 1):
        # Compute mean
        model = generate_model_to_compute_mean_and_var(channel_sizes, var_path, std_path, pca_model=pca_model)
        model.cuda()
        m = []
        for j in tqdm(range(0, subset_size, bs), desc=f"Layer {layerid} mean"):
            batch_input = data_array_combined[j:j+bs].cuda()
            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = torch.mean(output, axis=(2,3))
            output = output.cpu()
            m.append(output)
        m = torch.cat(m, dim=0)
        m = torch.mean(m, dim=0)
        m = m.view(1, len(m), 1, 1)
        m = m.cuda()

        # Compute variance
        std = []
        for j in tqdm(range(0, subset_size, bs), desc=f"Layer {layerid} var"):
            batch_input = data_array_combined[j:j+bs].cuda()
            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = output - m
            output = output ** 2
            output = torch.mean(output, axis=(2,3))
            output = output.cpu()
            std.append(output)
        std = torch.cat(std, dim=0)
        std = torch.mean(std, dim=0)
        m = m.cpu().squeeze()
        std = std.cpu()

        np.save(os.path.join(std_path, f'mean{layerid}.npy'), m.numpy())
        np.save(os.path.join(std_path, f'var{layerid}.npy'), std.numpy())

        # Generate model for projection matrix and bias computation
        model = generate_model_to_compute_pm_and_bias(channel_sizes, var_path, std_path, pca_model=pca_model)
        model.cuda()
        coefs = []
        for j in tqdm(range(0, subset_size, bs), desc=f"Layer {layerid} PCA"):
            batch_input = data_array_combined[j:j+bs].cuda()
            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = torch.mean(output, axis=(2,3))
            output = output.cpu()
            coefs.append(output)
        coefs = torch.cat(coefs, dim=0)

        # Mean centering
        channel_mean = torch.mean(coefs, axis=0)
        coefs = coefs - channel_mean

        if pca_model:
            pca = IncrementalPCA()
            pca.fit(coefs)
            projection_matrix = pca.components_
            projection_matrix = torch.tensor(projection_matrix, dtype=torch.float32)
            bias = projection_matrix @ channel_mean
            bias = -1 * bias
        else:
            num_features = coefs.shape[1]
            projection_matrix = torch.randn(num_features, num_features, dtype=torch.float32)
            bias = torch.zeros(num_features, dtype=torch.float32)

        np.save(os.path.join(var_path, f'pm_cent{layerid}.npy'), projection_matrix.numpy())
        np.save(os.path.join(var_path, f'bias{layerid}.npy'), bias.numpy())

        print(f"Layer {layerid} completed.")

def borderless_indices_in_loader(data_loader, border_width=1, threshold=0.25):
    borderless_indices = []
    subset = data_loader.dataset
    true_indices = subset.indices if isinstance(subset, Subset) else list(range(len(subset)))
    running_idx = 0
    for batch in tqdm(data_loader, desc="Scanning for borderless images"):
        images, _ = batch
        batch_size = images.shape[0]
        images_np = images.permute(0, 2, 3, 1).numpy()
        for i in range(batch_size):
            img_np = images_np[i]
            h, w, _ = img_np.shape
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            # Border extraction
            top = img_np[:border_width, :, :]
            bottom = img_np[-border_width:, :, :]
            left = img_np[:, :border_width, :]
            right = img_np[:, -border_width:, :]
            is_uniform = all([
                np.all(top == top[0,0,:]),
                np.all(bottom == bottom[0,0,:]),
                np.all(left == left[0,0,:]),
                np.all(right == right[0,0,:])
            ])
            if is_uniform:
                border_color = img_np[0, 0, :]
                matches = np.all(img_np == border_color, axis=-1)
                fraction = np.sum(matches) / (h * w)
                has_border = fraction <= threshold
                has_border = True # STRICT CONDITION
            else:
                has_border = False
            if not has_border:
                borderless_indices.append(true_indices[running_idx + i])
        running_idx += batch_size
    return borderless_indices

def train_model_borderless_sampled(
    channel_sizes, num_training_images, var_path, std_path, pca_model=True, seed=42, scan_size=20000):

    import os
    import random
    import torch
    import numpy as np
    from torch.utils.data import DataLoader, Subset
    from torchvision.datasets import ImageFolder
    from torchvision import transforms
    from torch.autograd import Variable
    from tqdm import tqdm

    # <-- Copy in your other needed imports here

    def borderless_indices_in_loader(data_loader, border_width=1, threshold=0.25):
        borderless_indices = []
        subset = data_loader.dataset
        true_indices = subset.indices if isinstance(subset, Subset) else list(range(len(subset)))
        running_idx = 0
        for batch in tqdm(data_loader, desc="Scanning for borderless images"):
            images, _ = batch
            batch_size = images.shape[0]
            images_np = images.permute(0, 2, 3, 1).numpy()
            for i in range(batch_size):
                img_np = images_np[i]
                h, w, _ = img_np.shape
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                top = img_np[:border_width, :, :]
                bottom = img_np[-border_width:, :, :]
                left = img_np[:, :border_width, :]
                right = img_np[:, -border_width:, :]
                is_uniform = all([
                    np.all(top == top[0,0,:]),
                    np.all(bottom == bottom[0,0,:]),
                    np.all(left == left[0,0,:]),
                    np.all(right == right[0,0,:])
                ])
                if is_uniform:
                    border_color = img_np[0, 0, :]
                    matches = np.all(img_np == border_color, axis=-1)
                    fraction = np.sum(matches) / (h * w)
                    has_border = fraction <= threshold
                    has_border = True # STRICT, as in your code
                else:
                    has_border = False
                if not has_border:
                    # Save the index in the *original* dataset (via subset.indices)
                    borderless_indices.append(true_indices[running_idx + i])
            running_idx += batch_size
        return borderless_indices

    # ========== Main process ==========

    if not os.path.exists(var_path):
        os.makedirs(var_path)
    if not os.path.exists(std_path):
        os.makedirs(std_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    random.seed(seed)
    torch.manual_seed(seed)
    folder_path = "/data/shared/datasets/imagenet"
    dataset = ImageFolder(root=folder_path, transform=transform)
    dataset_length = len(dataset)
    bs = 30

    # New Step 1: Pick 50,000 random indices from the dataset
    scan_indices = random.sample(range(dataset_length), min(scan_size, dataset_length))
    scan_subset = Subset(dataset, scan_indices)
    scan_loader = DataLoader(scan_subset, batch_size=bs, shuffle=False)

    # Step 2: Find borderless images among those 50,000
    print(f"Scanning {len(scan_indices)} random images for borderless images...")
    borderless_indices = borderless_indices_in_loader(scan_loader, border_width=1, threshold=1)
    print(f"Borderless images found: {len(borderless_indices)}/{len(scan_indices)}")

    if len(borderless_indices) < num_training_images:
        raise ValueError(
            f"Requested {num_training_images} training images, but only "
            f"{len(borderless_indices)} borderless images among scanned {len(scan_indices)} entries."
        )

    # Step 3: Randomly sample desired number of training indices from the borderless set
    subset_indices = random.sample(borderless_indices, num_training_images)
    borderless_subset = Subset(dataset, subset_indices)
    data_loader = DataLoader(borderless_subset, batch_size=bs, shuffle=True)

    # Step 4: The rest is as in your training function!
    for layerid in range(1, len(channel_sizes) + 1):
        model = generate_model_to_compute_mean_and_var(channel_sizes, var_path, std_path, pca_model=pca_model)        
        model.cuda()
        m = []
        j = 1
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / num_training_images) * 100, 2), end="\r")
            j += bs
            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            m.append(output)
        m = torch.cat(m, dim=0)
        m = torch.mean(m, dim=0)
        m = m.view(1, len(m), 1, 1)
        m = m.cuda()

        std = []
        j = 0
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / num_training_images) * 100, 2), end="\r")
            j += bs
            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = output - m
            output = output ** 2
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            std.append(output)

        std = torch.cat(std, dim=0)
        std = torch.mean(std, dim=0)
        m = m.cpu().squeeze()
        std = std.cpu()

        np.save(os.path.join(std_path, f'mean{layerid}.npy'), m.numpy())
        np.save(os.path.join(std_path, f'var{layerid}.npy'), std.numpy())

        # Generate model for projection matrix and bias computation
        model = generate_model_to_compute_pm_and_bias(channel_sizes, var_path, std_path, pca_model=pca_model)
        model.cuda()
        coefs = []
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / num_training_images) * 100, 2), end="\r")
            j += bs
            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            coefs.append(output)

        coefs = torch.cat(coefs, dim=0)
        channel_mean = torch.mean(coefs, axis=0)
        coefs = coefs - channel_mean

        if pca_model:
            from sklearn.decomposition import IncrementalPCA
            pca = IncrementalPCA()
            pca.fit(coefs)
            projection_matrix = pca.components_
            projection_matrix = torch.tensor(projection_matrix, dtype=torch.float32)
            bias = projection_matrix @ channel_mean
            bias = -1 * bias
        else:
            num_features = coefs.shape[1]
            projection_matrix = torch.randn(num_features, num_features, dtype=torch.float32)
            bias = torch.zeros(num_features, dtype=torch.float32)

        np.save(os.path.join(var_path, f'pm_cent{layerid}.npy'), projection_matrix.numpy())
        np.save(os.path.join(var_path, f'bias{layerid}.npy'), bias.numpy())

        print(f"Layer {layerid} completed.")

def train_model_pixelwise_pca(
    channel_sizes, num_training_images, var_path, std_path, pca_model=True, seed=42
):

    # Make directories
    if not os.path.exists(var_path):
        os.makedirs(var_path)
    if not os.path.exists(std_path):
        os.makedirs(std_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    random.seed(seed)
    torch.manual_seed(seed)
    folder_path = "/data/shared/datasets/imagenet"
    dataset = ImageFolder(root=folder_path, transform=transform)
    dataset_length = len(dataset)
    subset_size = num_training_images
    bs = 30
    subset_indices = random.sample(range(dataset_length), subset_size)
    dataset = Subset(dataset, subset_indices)
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True)

    for layerid in range(1, len(channel_sizes) + 1):
        # Mean and variance computation as before (pooled per channel)
        model = generate_model_to_compute_mean_and_var(
            channel_sizes, var_path, std_path, pca_model=pca_model
        )
        model.cuda()
        m = []
        j = 1
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs
            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            m.append(output)

        m = torch.cat(m, dim=0)
        m = torch.mean(m, dim=0)
        m = m.view(1, len(m), 1, 1)
        m = m.cuda()

        std = []
        j = 0
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs
            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = output - m
            output = output ** 2
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            std.append(output)

        std = torch.cat(std, dim=0)
        std = torch.mean(std, dim=0)
        m = m.cpu().squeeze()
        std = std.cpu()

        np.save(os.path.join(std_path, f'mean{layerid}.npy'), m.numpy())
        np.save(os.path.join(std_path, f'var{layerid}.npy'), std.numpy())

        # PCA and bias computation on every 10th pixel across all images (columns: channels)
        model = generate_model_to_compute_pm_and_bias(
            channel_sizes, var_path, std_path, pca_model=pca_model
        )
        model.cuda()
        coefs = []
        j = 0
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs

            with torch.no_grad():
                output = model(batch_input)  # [bs, nchannels, H, W]
            output = output.full_view()
            # Move channels to last: [bs, H, W, nchannels]
            output = output.permute(0, 2, 3, 1)
            # Flatten batch and spatial dims to pixels: [bs*H*W, nchannels]
            output = output.cpu().reshape(-1, output.size(-1))
            # Take every 10th pixel/pixel-location
            output = output[::10]
            coefs.append(output)

        # Concatenate all batches: shape [nimages*H*W/10, nchannels]
        coefs = torch.cat(coefs, dim=0)

        # Mean centering (per channel)
        channel_mean = torch.mean(coefs, axis=0)
        coefs = coefs - channel_mean

        if pca_model:
            # Compute PCA (features: channels; samples: pixels across all images)
            pca = IncrementalPCA()
            pca.fit(coefs)
            projection_matrix = pca.components_  # shape: [n_components, nchannels]
            projection_matrix = torch.tensor(projection_matrix, dtype=torch.float32)
            bias = projection_matrix @ channel_mean
            bias = -1 * bias
        else:
            num_features = coefs.shape[1]
            projection_matrix = torch.randn(num_features, num_features, dtype=torch.float32)
            bias = torch.zeros(num_features, dtype=torch.float32)

        np.save(os.path.join(var_path, f'pm_cent{layerid}.npy'), projection_matrix.numpy())
        np.save(os.path.join(var_path, f'bias{layerid}.npy'), bias.numpy())

        print(f"Layer {layerid} completed.")
        
# def train_control_model(channel_sizes, num_training_images, var_path, std_path, pca_model=True, seed=42):

#     if not os.path.exists(var_path):
#         os.makedirs(var_path)

#     if not os.path.exists(std_path):
#         os.makedirs(std_path)

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ])

#     random.seed(seed)
#     torch.manual_seed(seed)
#     folder_path = "/data/shared/datasets/imagenet"
#     dataset = ImageFolder(root=folder_path, transform=transform)
#     dataset_length = len(dataset)
#     subset_size = num_training_images
#     bs = 30
#     subset_indices = random.sample(range(dataset_length), subset_size)
#     dataset = Subset(dataset, subset_indices)
#     data_loader = DataLoader(dataset, batch_size=bs, shuffle=True)

#     for layerid in range(1, len(channel_sizes) + 1):
#         # Pass the pca_model flag to use the proper model generation function
#         model = generate_model_to_compute_mean_and_var(channel_sizes, var_path, std_path, pca_model=pca_model)
#         model.cuda()
#         m = []
#         j = 1
#         for batch_input, _ in data_loader:
#             batch_input = Variable(batch_input.cuda())
#             print(round((j / subset_size) * 100, 2), end="\r")
#             j += bs
#             with torch.no_grad():
#                 output = model(batch_input)
#                 output = output.full_view()
#                 output = torch.mean(output, axis=(2, 3))
#                 output = output.cpu()
#                 m.append(output)

#         m = torch.cat(m, dim=0)
#         m = torch.mean(m, dim=0)
#         m = m.view(1, len(m), 1, 1)
#         m = m.cuda()

#         std = []
#         j = 0
#         for batch_input, _ in data_loader:
#             batch_input = Variable(batch_input.cuda())
#             print(round((j / subset_size) * 100, 2), end="\r")
#             j += bs
#             with torch.no_grad():
#                 output = model(batch_input)
#                 output = output.full_view()
#                 output = output - m
#                 output = output ** 2
#                 output = torch.mean(output, axis=(2, 3))
#                 output = output.cpu()
#                 std.append(output)

#         std = torch.cat(std, dim=0)
#         std = torch.mean(std, dim=0)
#         m = m.cpu().squeeze()
#         std = std.cpu()

#         np.save(os.path.join(std_path, f'mean{layerid}.npy'), m.numpy())
#         np.save(os.path.join(std_path, f'var{layerid}.npy'), std.numpy())

#         # Generate model for projection matrix and bias computation
#         model = generate_model_to_compute_pm_and_bias(channel_sizes, var_path, std_path, pca_model=pca_model)
#         model.cuda()

#         coefs = []
#         for batch_input, _ in data_loader:
#             batch_input = Variable(batch_input.cuda())
#             print(round((j / subset_size) * 100, 2), end="\r")
#             j += bs

#             with torch.no_grad():
#                 output = model(batch_input)
#                 output = output.full_view()
#                 output = torch.mean(output, axis=(2, 3))
#                 output = output.cpu()
#                 coefs.append(output)

#         coefs = torch.cat(coefs, dim=0)

#         # Mean centering
#         channel_mean = torch.mean(coefs, axis=0)
#         coefs = coefs - channel_mean

#         if pca_model:
#             # Compute PCA on the coefficients
#             pca = IncrementalPCA()
#             pca.fit(coefs)
#             # Get the projection matrix and bias from PCA
#             projection_matrix = pca.components_
#             projection_matrix = torch.tensor(projection_matrix, dtype=torch.float32)
#             bias = projection_matrix @ channel_mean
#             bias = -1 * bias

#             # Flip the projection_matrix and bias along the first dimension
#             projection_matrix = torch.flip(projection_matrix, dims=[0])
#             bias = torch.flip(bias, dims=[0])

#         np.save(os.path.join(var_path, f'pm_cent{layerid}.npy'), projection_matrix.numpy())
#         np.save(os.path.join(var_path, f'bias{layerid}.npy'), bias.numpy())

#         print(f"Layer {layerid} completed.")

# def train_model_with_flops(channel_sizes, num_training_images, var_path, std_path, pca_model=True, seed=42):
#     # Create directories if they don't exist
#     if not os.path.exists(var_path):
#         os.makedirs(var_path)
#     if not os.path.exists(std_path):
#         os.makedirs(std_path)

#     # Define image transformations
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ])

#     # Set random seeds for reproducibility
#     random.seed(seed)
#     torch.manual_seed(seed)

#     # Load ImageNet dataset subset
#     folder_path = "/data/shared/datasets/imagenet"
#     dataset = ImageFolder(root=folder_path, transform=transform)
#     dataset_length = len(dataset)
#     subset_size = num_training_images
#     bs = 30
#     subset_indices = random.sample(range(dataset_length), subset_size)
#     dataset = Subset(dataset, subset_indices)
#     data_loader = DataLoader(dataset, batch_size=bs, shuffle=True)

#     # Loop over each layer
#     for layerid in range(1, len(channel_sizes) + 1):
#         # Model to compute mean and variance
#         model = generate_model_to_compute_mean_and_var(channel_sizes, var_path, std_path, pca_model=pca_model)
#         # Compute FLOPs and parameters
#         with torch.cuda.device(0):
#             macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True)
#         print(f"Layer {layerid} mean_var model FLOPs: {macs}, Params: {params}")
#         model.cuda()

#         # Compute mean activations
#         m = []
#         j = 1
#         for batch_input, _ in data_loader:
#             batch_input = Variable(batch_input.cuda())
#             print(round((j / subset_size) * 100, 2), end="\r")
#             j += bs
#             with torch.no_grad():
#                 output = model(batch_input)
#             output = output.full_view()
#             output = torch.mean(output, axis=(2, 3))
#             output = output.cpu()
#             m.append(output)

#         m = torch.cat(m, dim=0)
#         m = torch.mean(m, dim=0)
#         m = m.view(1, len(m), 1, 1).cuda()

#         # Compute variance of activations
#         std = []
#         j = 0
#         for batch_input, _ in data_loader:
#             batch_input = Variable(batch_input.cuda())
#             print(round((j / subset_size) * 100, 2), end="\r")
#             j += bs
#             with torch.no_grad():
#                 output = model(batch_input)
#             output = output.full_view()
#             output = output - m
#             output = output ** 2
#             output = torch.mean(output, axis=(2, 3))
#             output = output.cpu()
#             std.append(output)

#         std = torch.cat(std, dim=0)
#         std = torch.mean(std, dim=0)
#         m = m.cpu().squeeze()
#         std = std.cpu()

#         # Save mean and variance
#         np.save(os.path.join(std_path, f'mean{layerid}.npy'), m.numpy())
#         np.save(os.path.join(std_path, f'var{layerid}.npy'), std.numpy())

#         # Model to compute projection matrix and bias
#         model = generate_model_to_compute_pm_and_bias(channel_sizes, var_path, std_path, pca_model=pca_model)
#         # Compute FLOPs and parameters for PM/Bias model
#         with torch.cuda.device(0):
#             macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True)
#         print(f"Layer {layerid} pm_bias model FLOPs: {macs}, Params: {params}")
#         model.cuda()

#         # Compute coefficients
#         coefs = []
#         j = 0
#         for batch_input, _ in data_loader:
#             batch_input = Variable(batch_input.cuda())
#             print(round((j / subset_size) * 100, 2), end="\r")
#             j += bs
#             with torch.no_grad():
#                 output = model(batch_input)
#             output = output.full_view()
#             output = torch.mean(output, axis=(2, 3))
#             output = output.cpu()
#             coefs.append(output)

#         coefs = torch.cat(coefs, dim=0)

#         # Mean centering
#         channel_mean = torch.mean(coefs, axis=0)
#         coefs = coefs - channel_mean

#         # Compute PCA on the coefficients
#         pca = IncrementalPCA()
#         pca.fit(coefs)
#         # Get the projection matrix and bias from PCA
#         projection_matrix = pca.components_
#         projection_matrix = torch.tensor(projection_matrix, dtype=torch.float32)
#         bias = projection_matrix @ channel_mean
#         bias = -1 * bias
        
#         np.save(os.path.join(var_path, f'pm_cent{layerid}.npy'), projection_matrix.numpy())
#         np.save(os.path.join(var_path, f'bias{layerid}.npy'), bias.numpy())

#         print(f"Layer {layerid} completed.")
# import os
# import numpy as np
# import torch.nn as nn
# from main_new_OG import *
# import torch
# from torch.utils.data import DataLoader, Subset
# from torchvision import transforms
# from torchvision.datasets import ImageFolder
# from torch.autograd import Variable
# import numpy as np
# import random
# from sklearn.decomposition import IncrementalPCA
# from main import *
# import h5py
# import xarray as xr
# from torch_cv import *
# from regression import *
# from sklearn.linear_model import Ridge
# import shutil

# def generate_model_to_compute_mean_and_var(num_channels, var_path, std_path):
        
#     model = load_model_modified(num_channels)
#     n = len([f for f in os.listdir(var_path) if f.startswith('bias')])
    
#     if n > 0: 
#         new_num_channels = num_channels[:n]
#         m1 = make_pca(new_num_channels, var_path, std_path)
        
#         new_model = nn.Sequential(
#             model[0], 
#             *[m1[i] for i in range(1, len(new_num_channels) + 1)], 
#             model[len(new_num_channels) + 1][:2]
#             )
#     else: 
#         new_model = nn.Sequential(
#             model[0], 
#             model[1][:2]
#             )
    
#     return new_model

# def generate_model_to_compute_pm_and_bias(num_channels, var_path, std_path):
#     n = len([f for f in os.listdir(std_path) if f.startswith('mean')])
#     if n % 2 == 1:
#         model = load_model_new(num_channels[:n], std_path)        
#         if n > 1:
#             m1 = make_pca(num_channels[:n-1], var_path, std_path)
#             new_model = nn.Sequential(
#                 model[0],  # First layer from model
#                 *[m1[i] for i in range(1, len(num_channels[:n-1]) + 1)], 
#                 model[len(num_channels[:n-1]) + 1][:3]  # Final layers from model
#             )
#         else:
#             new_model = nn.Sequential(
#                 model[0],  # First layer from model
#                 model[1][:3]  # Final layers from model
#             )
#     else:
#         dummy_mean_path = os.path.join(std_path, f'mean{n+1}.npy')
#         dummy_var_path = os.path.join(std_path, f'var{n+1}.npy')
#         np.save(dummy_mean_path, np.random.randn(1))  # Create dummy mean file
#         np.save(dummy_var_path, np.random.randn(1))  # Create dummy variance file
#         model = load_model_new(num_channels[:n+1], std_path)
#         m1 = make_pca(num_channels[:n-1], var_path, std_path)
#         new_model = nn.Sequential(
#             model[0],  # First layer from model
#             *[m1[i] for i in range(1, len(num_channels[:n-1]) + 1)],  # Layers from m1
#             model[len(num_channels[:n-1]) + 1][:3]  # Final layers from model
#         )
#         os.remove(dummy_mean_path)
#         os.remove(dummy_var_path)      
#     return new_model


def train_model_k(
    channel_sizes, num_training_images, var_path, std_path,
    k, pca_model=True, seed=42
):
    """
    Train model as before, but store projection matrices with top k PCA eigenvectors
    and random rows for the rest.

    Args:
        channel_sizes: list[int] -- feature dims per layer
        num_training_images: int -- number of images for stats
        var_path, std_path: directories for saving stats
        k: int -- number of PCA eigendirections to keep (rest are random)
        pca_model: bool -- if True, do PCA; if False, all random
        seed: int
    """
    if not os.path.exists(var_path):
        os.makedirs(var_path)

    if not os.path.exists(std_path):
        os.makedirs(std_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    random.seed(seed)
    torch.manual_seed(seed)
    folder_path = "/data/shared/datasets/imagenet"
    dataset = ImageFolder(root=folder_path, transform=transform)
    dataset_length = len(dataset)
    subset_size = num_training_images
    bs = 30
    subset_indices = random.sample(range(dataset_length), subset_size)
    dataset = Subset(dataset, subset_indices)
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True)

    for layerid in range(1, len(channel_sizes) + 1):
        model = generate_model_to_compute_mean_and_var(
            channel_sizes, var_path, std_path, pca_model=pca_model
        )
        model.cuda()
        m = []
        j = 1
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs
            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            m.append(output)

        m = torch.cat(m, dim=0)
        m = torch.mean(m, dim=0)
        m = m.view(1, len(m), 1, 1)
        m = m.cuda()

        std = []
        j = 0
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs
            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = output - m
            output = output ** 2
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            std.append(output)

        std = torch.cat(std, dim=0)
        std = torch.mean(std, dim=0)
        m = m.cpu().squeeze()
        std = std.cpu()

        np.save(os.path.join(std_path, f'mean{layerid}.npy'), m.numpy())
        np.save(os.path.join(std_path, f'var{layerid}.npy'), std.numpy())

        # Now, for projection matrix and bias
        model = generate_model_to_compute_pm_and_bias(
            channel_sizes, var_path, std_path, pca_model=pca_model
        )
        model.cuda()
        coefs = []
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs

            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            coefs.append(output)

        coefs = torch.cat(coefs, dim=0)

        # Mean centering
        channel_mean = torch.mean(coefs, axis=0)
        coefs = coefs - channel_mean

        num_features = coefs.shape[1]
        if k > num_features:
            raise ValueError(f"k = {k} must be <= number of channels ({num_features})")

        if pca_model:
            # Compute PCA
            pca = IncrementalPCA(n_components=min(num_features, coefs.shape[0]))
            pca.fit(coefs)
            # Get projection (full)
            projection_matrix_full = torch.tensor(pca.components_, dtype=torch.float32)
            # Get top k
            projection_matrix_topk = projection_matrix_full[:k]
            # For rest, random
            if k < num_features:
                projection_rand = torch.randn(num_features-k, num_features)
                # Optionally, make sure random rows are orthogonal to PCA space (optional)
                # Or just orthonormalize all together.
                projection_rand = torch.nn.functional.normalize(projection_rand, dim=1)
                # Combine
                projection_matrix = torch.cat([projection_matrix_topk, projection_rand], dim=0)
            else:
                projection_matrix = projection_matrix_topk
            # Bias for top k
            bias = projection_matrix @ channel_mean
            bias = -1 * bias

        else:
            # Completely random orthogonal matrix (for repeatability: use torch.manual_seed above)
            projection_matrix = torch.randn(num_features, num_features, dtype=torch.float32)
            # Optionally orthogonalize (Gram-Schmidt or QR)
            projection_matrix, _ = torch.linalg.qr(projection_matrix)
            bias = torch.zeros(num_features, dtype=torch.float32)

        np.save(os.path.join(var_path, f'pm_cent{layerid}.npy'), projection_matrix.numpy())
        np.save(os.path.join(var_path, f'bias{layerid}.npy'), bias.numpy())

        print(f"Layer {layerid} completed.")
        
def train_model_ica(channel_sizes, num_training_images, var_path, std_path, ica_model=True, seed=42, max_iter=3):

    if not os.path.exists(var_path):
        os.makedirs(var_path)
        
    if not os.path.exists(std_path):
        os.makedirs(std_path)    

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    folder_path = "/data/shared/datasets/imagenet"
    dataset = ImageFolder(root=folder_path, transform=transform)
    dataset_length = len(dataset)
    subset_size = num_training_images
    bs = 30
    subset_indices = random.sample(range(dataset_length), subset_size)
    dataset = Subset(dataset, subset_indices)
    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True)

    for layerid in range(1, len(channel_sizes) + 1):
        # Use the same model generation functions, passing ica_model as pca_model
        # (they work the same way - just loading/saving projection matrices)
        model = generate_model_to_compute_mean_and_var(channel_sizes, var_path, std_path, pca_model=ica_model)        
        model.cuda()
        m = []
        j = 1
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs
            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            m.append(output)

        m = torch.cat(m, dim=0)
        m = torch.mean(m, dim=0)
        m = m.view(1, len(m), 1, 1)
        m = m.cuda()

        std = []
        j = 0
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs
            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = output - m
            output = output ** 2
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            std.append(output)

        std = torch.cat(std, dim=0)
        std = torch.mean(std, dim=0)
        m = m.cpu().squeeze()
        std = std.cpu()

        np.save(os.path.join(std_path, f'mean{layerid}.npy'), m.numpy())
        np.save(os.path.join(std_path, f'var{layerid}.npy'), std.numpy())

        # Generate model for projection matrix and bias computation
        model = generate_model_to_compute_pm_and_bias(channel_sizes, var_path, std_path, pca_model=ica_model)
        model.cuda()
        coefs = []
        for batch_input, _ in data_loader:
            batch_input = Variable(batch_input.cuda())
            print(round((j / subset_size) * 100, 2), end="\r")
            j += bs

            with torch.no_grad():
                output = model(batch_input)
            output = output.full_view()
            output = torch.mean(output, axis=(2, 3))
            output = output.cpu()
            coefs.append(output)

        coefs = torch.cat(coefs, dim=0)

        # Mean centering
        channel_mean = torch.mean(coefs, axis=0)
        coefs = coefs - channel_mean

        if ica_model:
            # Compute ICA on the coefficients
            n_components = coefs.shape[1]
            ica = FastICA(
                n_components=n_components, 
                random_state=seed, 
                max_iter=max_iter, 
                whiten='unit-variance',
                tol=1e-4
            )
            try:
                ica.fit(coefs.numpy())
                # Get the unmixing matrix (projection matrix) from ICA
                # The components_ attribute contains the unmixing matrix
                projection_matrix = ica.components_  # Shape: (n_components, n_features)
                projection_matrix = torch.tensor(projection_matrix, dtype=torch.float32)
            except Exception as e:
                print(f"Warning: ICA did not converge for layer {layerid}. Error: {e}")
                print("Falling back to random orthogonal matrix.")
                num_features = coefs.shape[1]
                projection_matrix = torch.tensor(
                    ortho_group.rvs(dim=num_features).astype(np.float32)
                )
            
            bias = projection_matrix @ channel_mean
            bias = -1 * bias
        else:
            # For random models, generate a random projection matrix and use zero bias
            num_features = coefs.shape[1]
            projection_matrix = torch.randn(num_features, num_features, dtype=torch.float32)
            bias = torch.zeros(num_features, dtype=torch.float32)

        np.save(os.path.join(var_path, f'pm_cent{layerid}.npy'), projection_matrix.numpy())
        np.save(os.path.join(var_path, f'bias{layerid}.npy'), bias.numpy())

        print(f"Layer {layerid} completed.")