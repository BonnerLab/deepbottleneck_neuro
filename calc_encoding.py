import os
import sys
sys.path.append('/home/robinbs1/cortical_transformers/deepbottleneck')
sys.path.append('/home/robinbs1/cortical_transformers/deepbottleneck/helpers')
import pandas as pd
from transformers import AutoModel, AutoConfig
from training import PcaModel, PcaModelConfig #necessary to register the model...
from transformers.trainer_utils import get_last_checkpoint

import torch
import numpy as np
from bonner.models.hooks import compute_johnson_lindenstrauss_limit, SparseRandomProjection
from datetime import datetime
import json
from copy import deepcopy as cp
from main import load_model
from helpers.initialise_models import *
from transformers import ResNetForImageClassification
from PIL import Image
from torchvision import transforms

from dotenv import load_dotenv
load_dotenv()

# NSD_DATA_PATH=os.getenv('NSD_DATA_PATH')
NSD_DATA_PATH = os.environ['NSD_DATA_PATH']
# for_atlas_path = '/home/robinbs1/cortical_transformers/deep_bottleneck_data/for_atlas'
NSD_STIMULI_PATH = os.environ['NSD_STIMULI_PATH']
# hdf5_file_path = '/home/robinbs1/cortical_transformers/deep_bottleneck_data/nsd_stimuli.hdf5'
BACKPROP_TRAINING_PATH=os.environ['BACKPROP_TRAINING_PATH']
# training_run_folder='/home/robinbs1/cortical_transformers/deep_bottleneck_data/bottleneck_training_runs'
ENCODING_EVAL_PATH = os.environ['ENCODING_EVAL_PATH']
# encoding_eval_folder = '/home/robinbs1/cortical_transformers/deep_bottleneck_data/enocding_eval'
ENCODING_CACHE_PATH=os.environ['ENCODING_CACHE_PATH']
PRETRAINED_SCATTERING_PATH=os.environ['PRETRAINED_SCATTERING_PATH']


from helpers.encoding_score_tools_new import (select_imageids, get_neural_activations, load_images, extract_activations,
                                      compute_encoding_score, compute_rsa, compute_srp)
from helpers.encoding_score_monkey import *


def quick_load_preproc_images(subjid,device='cpu'):
    fol_preproc_im_save = f'{ENCODING_CACHE_PATH}/images/subj{subjid}_split'
    os.makedirs(fol_preproc_im_save,exist_ok=True)
    f_unshared = os.path.join(fol_preproc_im_save,'data_array_unshared.pt')
    f_shared =  os.path.join(fol_preproc_im_save,'data_array_shared.pt')
    if os.path.exists(f_shared) and os.path.exists(f_unshared):
         print(f"loading {f_shared}, {f_shared}")
         data_array_unshared, data_array_shared = torch.load(f_unshared).to(device), torch.load(f_shared).to(device)
    else:
        data_array_unshared, data_array_shared = load_images(subjid,
                                                            nsd_data_path=NSD_DATA_PATH,
                                                            hdf5_file_path = NSD_STIMULI_PATH,
                                                            device=device)
        torch.save(data_array_unshared, f_unshared)
        torch.save(data_array_shared, f_shared)
        print(f"saved out {f_shared}, {f_unshared}")
       
    return data_array_unshared, data_array_shared

def quick_load_proc_neural_data(subjid,roi):
    fol_preproc_neural_save = f'{ENCODING_CACHE_PATH}/pre_proc_neural/{roi}_subj{subjid}_split'
    
    f_y_train = os.path.join(fol_preproc_neural_save,'y_train.npy')
    f_y_test = os.path.join(fol_preproc_neural_save,'y_test.npy')
    if os.path.exists(f_y_train) and os.path.exists(f_y_test):
        print(f"loading {f_y_train}, {f_y_test}")
        ytrain, ytest = np.load(f_y_train), np.load(f_y_test)
    else:
        os.makedirs(fol_preproc_neural_save,exist_ok=True)
        ytrain, ytest = get_neural_activations(subjid, roi,nsd_data_path=NSD_DATA_PATH)
        os.makedirs(fol_preproc_neural_save, exist_ok=True)
        np.save(f_y_train,ytrain)
        np.save(f_y_test,ytest)
        print(f"saved out {f_y_train}, {f_y_test}")
    
        
    return ytrain, ytest

def quick_load_things_images_and_embeddings(
    cache_folder='/data/shared/datasets/hebart2022.things.behavior/cache',
    base_folder='/data/shared/datasets/hebart2019.things/images/object_images',
    order_file='/data/shared/datasets/hebart2022.things.behavior/variables/unique_id.txt',
    embedding_file='/data/shared/datasets/hebart2022.things.behavior/data/spose_embedding_66d_sorted.txt',
    device='cpu',
):
    os.makedirs(cache_folder, exist_ok=True)
    images_tensor_cache = os.path.join(cache_folder, 'things_images_tensor.pt')
    embeddings_cache = os.path.join(cache_folder, 'things_embeddings.npy')

    # Step 1: Load cached tensors/embeddings if available
    if os.path.exists(images_tensor_cache) and os.path.exists(embeddings_cache):
        print(f"Loading cached images and embeddings from {images_tensor_cache}, {embeddings_cache}")
        images_tensor = torch.load(images_tensor_cache).to(device)
        embeddings = np.load(embeddings_cache)
        return images_tensor, embeddings

    # Step 2: Load ordered folder names
    with open(order_file, 'r') as f:
        folder_order = [line.strip() for line in f if line.strip()]

    # Step 3: Load and preprocess images (this is expensive, so only do it if not cached)
    images = []
    for folder in folder_order:
        folder_path = os.path.join(base_folder, folder)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Can't find directory: {folder_path}")
        matches = [fname for fname in os.listdir(folder_path) if "01" in fname]
        if not matches:
            raise FileNotFoundError(f"No files with '01' in {folder_path}")
        img_path = os.path.join(folder_path, matches[0])
        img = Image.open(img_path).convert("RGB")
        images.append(img)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    tensors = [transform(img) for img in images]
    images_tensor = torch.stack(tensors)  # (nimages, channels, height, width)

    # Step 4: Load embeddings
    embeddings = np.loadtxt(embedding_file)

    # Step 5: Save to cache for next time
    torch.save(images_tensor.cpu(), images_tensor_cache)
    np.save(embeddings_cache, embeddings)
    print(f"Saved {images_tensor_cache} and {embeddings_cache}")

    # Move tensor to device if needed
    images_tensor = images_tensor.to(device)

    return images_tensor, embeddings


def calc_encoding_score(model,subjid,roi,layerid,device='cuda'):
    data_array_unshared, data_array_shared = quick_load_preproc_images(subjid,device=device)
    xtrain, xtest = extract_activations(model, layerid, data_array_unshared, data_array_shared)
    ytrain, ytest= quick_load_proc_neural_data(subjid,roi)
    e = compute_encoding_score(xtrain=xtrain,
        xtest = xtest,
        ytrain=ytrain,
        ytest=ytest,
        epsilon = 0.1,
        device=device)
    return e

def calc_embedding_score(model, layerid):
    images_tensor, embeddings = quick_load_things_images_and_embeddings()
    mmodel = model[:layerid + 1]
    x = extract_all_activations(images_tensor, mmodel)
    r2 = monkey_regression(x, embeddings).mean()
    return r2

# def get_model(model_label, model_type='trained_bottleneck', ckpt_epoch=0):
#     """Loads a model for encoding comparison.

#     For a trained resnet model, the saved configuration should have the attribute
#     'resnet_baseline' set to True so that we can load the appropriate model class.
    
#     Args:
#         model_label (str): subfolder after BACKPROP_TRAINING_PATH where model is finetuned.
#         model_type (str, optional): 'trained_bottleneck' or 'scattering'. Defaults to 'trained_bottleneck'.
#         ckpt_epoch (int, optional): Epoch checkpoint to load. 0 loads the latest checkpoint.
    
#     Returns:
#         model: The loaded model.
#     """
#     if model_type == 'trained_bottleneck':
#         model_training_folder = os.path.join(BACKPROP_TRAINING_PATH, model_label)
#         if ckpt_epoch > 0:
#             checkpoint_path = os.path.join(model_training_folder, f"checkpoint-{ckpt_epoch}")
#             config = AutoConfig.from_pretrained(checkpoint_path)
#             # Check if this is a resnet model
#             if hasattr(config, "resnet_baseline") and config.resnet_baseline:
#                 print()
#                 model = ResNetForImageClassification.from_pretrained(checkpoint_path)
#                 print("Loaded ResNet model from checkpoint.")
#             else:
#                 # For PCA model (or random model) loading
#                 model = PcaModel.from_pretrained(checkpoint_path)
#                 print("Loaded PCA model from checkpoint.")
#         else:
#             # If no specific epoch is provided, load from the last checkpoint.
#             last_checkpoint = get_last_checkpoint(model_training_folder)
#             config_path = os.path.join(last_checkpoint, 'config.json')
#             config = AutoConfig.from_pretrained(config_path)
#             if hasattr(config, "resnet_baseline") and config.resnet_baseline:
#                 from transformers import ResNetForImageClassification
#                 model = ResNetForImageClassification.from_config(config)
#                 print("Initialized ResNet model from configuration.")
#             else:
#                 model = PcaModel.from_config(config)
#                 print("Initialized PCA model from configuration.")
#     elif model_type == 'scattering':
#         path = os.path.join(PRETRAINED_SCATTERING_PATH, f'{model_label}.pth.tar')
#         model = load_model()
#         checkpoint = torch.load(path)
#         state_dict = checkpoint["state_dict"]
#         state_dict = {key.replace("(0, 0)", "0"): value for key, value in state_dict.items()}
#         checkpoint["state_dict"] = state_dict
#         model.load_state_dict(checkpoint['state_dict'])
#         print("Loaded scattering model.")
#     return model


def get_model(model_label,model_type='trained_bottleneck',ckpt_epoch=0):
    """loads a model for encoding comparison

    Args:
        model_label (str): subfolder after BACKPROP_TRAINING_PATH where model is finetuned (or see note below for scattering)
        model_type (str, optional): _description_. Defaults to 'trained_bottleneck'.
            other option is scattering, for scattering appropriate model_label is batchsize_128_lrfreq_45_best
        ckpt_epoch (int, optional): _description_. Defaults to 0.
            0 loads a re-initialized model

    Returns:
        _type_: _description_
    """
    if model_type == 'trained_bottleneck':
        model_training_folder = os.path.join(BACKPROP_TRAINING_PATH,model_label)
        if ckpt_epoch > 0:
            model = AutoModel.from_pretrained(os.path.join(model_training_folder,
                                                   f"checkpoint-{ckpt_epoch}")).model
        else:
            last_checkpoint = get_last_checkpoint(model_training_folder)
            config =  AutoConfig.from_pretrained(os.path.join(last_checkpoint,'config.json'))
            model = AutoModel.from_config(config).model
    elif model_type == 'scattering':
        # batchsize_128_lrfreq_45_best
        path = os.path.join(PRETRAINED_SCATTERING_PATH)
        model = load_model()
        checkpoint = torch.load(path)
        state_dict = checkpoint["state_dict"]
        state_dict = {key.replace("(0, 0)", "0"): value for key, value in state_dict.items()}
        checkpoint["state_dict"] = state_dict
        model.load_state_dict(checkpoint['state_dict'])
    return model

       
# def get_model(model_type='trained_bottleneck',
#                 model_label='2024_1203_scatt_lr1e-4',
#                 ckpt_epoch = 10010):
#     # batchsize_128_lrfreq_45_best
#     if model_type == 'trained_bottleneck':
#         if ckpt_epoch > 0:
#             checkpoint_path = f"{BACKPROP_TRAINING_PATH}/{model_label}/checkpoint-{ckpt_epoch}"
#             model = PcaModel.from_pretrained(checkpoint_path).model
#         else:
#             pca_config = PcaModelConfig()
#             model = PcaModel(pca_config).model
#     elif model_type == 'bp_scattering':
#         path = os.path.join(PRETRAINED_SCATTERING_PATH,f'.pth.tar'
#         model = load_model()
#         checkpoint = torch.load(path)
#         state_dict = checkpoint["state_dict"]
#         state_dict = {key.replace("(0, 0)", "0"): value for key, value in state_dict.items()}
#         checkpoint["state_dict"] = state_dict
#         model.load_state_dict(checkpoint['state_dict'])
#     elif model_type == 'random':
#         if ckpt_epoch > 0:
#             checkpoint_path = f"{BACKPROP_TRAINING_PATH}/{model_label}/checkpoint-{ckpt_epoch}"
#             model = PcaModel.from_pretrained(checkpoint_path).model
#         else:
#             pca_config = PcaModelConfig(random_init=True)
#             model = PcaModel(pca_config).model


#     return model

def check_for_duplicates(json_directory,reference_subset):
    matches_found = False
    for filename in os.listdir(json_directory):
        if filename.endswith(".json"):
            file_path = os.path.join(json_directory, filename)
            
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)  # load JSON as dict
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse JSON file {file_path}:\n{e}")
                    continue
                
                # Check the subset of fields
                # 'all' will be True only if data.get(k) == v for every (k, v) in our reference_subset
                if all(data.get(k) == v for k, v in reference_subset.items()):
                    print(f"Subset match found in {file_path}")
                    matches_found = True
    return matches_found


def run_encoding_eval(model_args,encoding_args,encoding_eval_label,device='cuda'):
    
    result_dict = cp(encoding_args)
    result_dict.update(model_args)
    save_folder = os.path.join(ENCODING_EVAL_PATH,encoding_eval_label)
    os.makedirs(save_folder, exist_ok=True)
    if not check_for_duplicates(save_folder,result_dict):
        model = get_model(**model_args)
        e = calc_encoding_score(model=model,
                                device=device,
                                **encoding_args)
        
        
        result_dict['encoding_score']=e.numpy().item()
        # for k in result_dict:
        #     print(f'{k}: {result_dict[k]}, class: {result_dict[k].__class__}')
        label_dict = cp(encoding_args)
        label_dict['ckpt_epoch']=model_args['ckpt_epoch']
        encoding_args_str = "__".join(f"{key}_{value}" for key, value in label_dict.items())
        
        timestamp = datetime.now().strftime("%Y_%m%d_%H%M%S")
        f_save=os.path.join(save_folder,f"{timestamp}_{encoding_args_str}.json")
        
        with open(f_save, "w") as json_file:
            json.dump(result_dict, json_file, indent=4)
            
def run_embedding_eval(model_args, encoding_eval_label, layerid, device='cuda'):
    result_dict = cp(model_args)
    result_dict['layerid'] = layerid
    save_folder = os.path.join(ENCODING_EVAL_PATH, encoding_eval_label)
    os.makedirs(save_folder, exist_ok=True)
    if not check_for_duplicates(save_folder, result_dict):
        model = get_model(**model_args)
        e = calc_embedding_score(model=model, layerid=layerid)

        result_dict['embedding_score'] = float(e)  # ensure serializable

        # Build label for JSON naming
        label_dict = cp(model_args)
        label_dict['layerid'] = layerid
        args_str = "__".join(f"{key}_{value}" for key, value in label_dict.items())

        timestamp = datetime.now().strftime("%Y_%m%d_%H%M%S")
        f_save = os.path.join(save_folder, f"{timestamp}_{args_str}.json")

        with open(f_save, "w") as json_file:
            json.dump(result_dict, json_file, indent=4)


def load_encoding_results(encoding_label):

    def load_acc(model_label,ckpt_epoch):
        f_load = f'{BACKPROP_TRAINING_PATH}/{model_label}/checkpoint-{ckpt_epoch}/trainer_state.json'
        with open(f_load,'r') as f:
            trainer_state = json.load(f)
            eval_accuracy=trainer_state['log_history'][-1]['eval_accuracy']
        return eval_accuracy


    eval_results_dir = os.path.join(ENCODING_EVAL_PATH,encoding_label)
    records = []
    for filename in os.listdir(eval_results_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(eval_results_dir, filename)
            
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            records.append(data)
    df=pd.DataFrame().from_records(records)
    eval_accuracy = []
    for i, row in df.iterrows():
        if (row['model_type'] in ['random','trained_bottleneck']) and (row['ckpt_epoch']>0):
            try:
                eval_accuracy.append(load_acc(row['model_label'],row['ckpt_epoch']))
            except:
                eval_accuracy.append(np.nan)
        else:
            eval_accuracy.append(np.nan)
        # print(row)
    df['eval_accuracy']=eval_accuracy

    return df

def calc_encoding_score_alexnet(subjid, roi, layerid, training_status=True, device='cuda'):

    data_array_unshared, data_array_shared = quick_load_preproc_images(subjid, device=device)
    
    xtrain = extract_alexnet_activations(data_array_unshared, training_status, layerid)
    xtest = extract_alexnet_activations(data_array_shared, training_status, layerid)
    
    ytrain, ytest = quick_load_proc_neural_data(subjid, roi)
    
    e = compute_encoding_score(
        xtrain=xtrain,
        xtest=xtest,
        ytrain=ytrain,
        ytest=ytest,
        epsilon=0.1,
        device=device
    )
    return e

def run_encoding_eval_alexnet(encoding_args, encoding_eval_label, device='cuda'):
    """
    Run encoding evaluation for AlexNet and save results.
    
    Args:
        encoding_args: Dictionary containing:
            - subjid: Subject ID
            - roi: Region of interest
            - layerid: Layer ID for feature extraction
            - training_status: Whether to use trained (True) or untrained (False) AlexNet
        encoding_eval_label: Label for the evaluation (used for save folder)
        device: Device to run computations on
    """
    result_dict = cp(encoding_args)
    result_dict['model_type'] = 'alexnet'
    
    save_folder = os.path.join(ENCODING_EVAL_PATH, encoding_eval_label)
    os.makedirs(save_folder, exist_ok=True)
    
    if not check_for_duplicates(save_folder, result_dict):
        e = calc_encoding_score_alexnet(
            subjid=encoding_args['subjid'],
            roi=encoding_args['roi'],
            layerid=encoding_args['layerid'],
            training_status=encoding_args.get('training_status', True),
            device=device
        )
        
        result_dict['encoding_score'] = e.numpy().item()
        
        # Create label string for filename
        label_dict = cp(encoding_args)
        encoding_args_str = "__".join(f"{key}_{value}" for key, value in label_dict.items())
        
        timestamp = datetime.now().strftime("%Y_%m%d_%H%M%S")
        f_save = os.path.join(save_folder, f"{timestamp}_{encoding_args_str}.json")
        
        with open(f_save, "w") as json_file:
            json.dump(result_dict, json_file, indent=4)
        
        print(f"Results saved to: {f_save}")
    else:
        print("Duplicate found, skipping evaluation.")

def calc_encoding_score_alexnet_general(
    subjid, roi,
    module, layer_idx,
    training_status=True,
    device='cuda'
):
    """
    module: 'features' or 'classifier'
    layer_idx: int -- index in the module
    """
    data_array_unshared, data_array_shared = quick_load_preproc_images(subjid, device=device)

    # Extract features using the generalized function
    xtrain = extract_layer_activations(
        data_array_unshared,
        trained=training_status,
        module=module,
        layer_idx=layer_idx,
        batch_size=50,  # or set in config
        gpool=False     # or True if you want, for conv layers
    )

    xtest = extract_layer_activations(
        data_array_shared,
        trained=training_status,
        module=module,
        layer_idx=layer_idx,
        batch_size=50,
        gpool=False
    )

    ytrain, ytest = quick_load_proc_neural_data(subjid, roi)
    e = compute_encoding_score(
        xtrain=xtrain,
        xtest=xtest,
        ytrain=ytrain,
        ytest=ytest,
        epsilon=0.1,
        device=device
    )
    return e

def run_encoding_eval_alexnet_general(encoding_args, encoding_eval_label, device='cuda'):
    """
    Run encoding evaluation for AlexNet (any layer) and save results.
    encoding_args should contain:
        - subjid
        - roi
        - module: 'features' or 'classifier'
        - layer_idx: int
        - training_status: True/False
    """

    result_dict = cp(encoding_args)
    result_dict['model_type'] = 'alexnet'

    save_folder = os.path.join(ENCODING_EVAL_PATH, encoding_eval_label)
    os.makedirs(save_folder, exist_ok=True)

    if not check_for_duplicates(save_folder, result_dict):
        e = calc_encoding_score_alexnet_general(
            subjid=encoding_args['subjid'],
            roi=encoding_args['roi'],
            module=encoding_args['module'],
            layer_idx=encoding_args['layer_idx'],
            training_status=encoding_args.get('training_status', True),
            device=device
        )

        result_dict['encoding_score'] = e.detach().cpu().item()

        # Create label string for filename
        label_dict = cp(encoding_args)
        encoding_args_str = "__".join(f"{key}_{value}" for key, value in label_dict.items())

        timestamp = datetime.now().strftime("%Y_%m%d_%H%M%S")
        f_save = os.path.join(save_folder, f"{timestamp}_{encoding_args_str}.json")

        with open(f_save, "w") as json_file:
            json.dump(result_dict, json_file, indent=4)

        print(f"Results saved to: {f_save}")
    else:
        print("Duplicate found, skipping evaluation.")
        
def calc_training_encoding_score(model, subjid, roi, layerid, device='cuda'):

    du, ds = quick_load_preproc_images(subjid, device=device)
    x, x_ = extract_activations(model, layerid, du, ds)
    du = None
    ds = None
    
    n = compute_johnson_lindenstrauss_limit(n_samples=x.shape[0], epsilon=0.1)     
    sparse_random_projection = SparseRandomProjection(
        n_components=n,
        density=None,
        seed=0,
        allow_expansion=False
    )
    
    x = sparse_random_projection(x).cpu().numpy()
    y, y_ = quick_load_proc_neural_data(subjid, roi)
    
    ALPHA_RANGE = [10**i for i in range(10)]    
    regression = TorchRidgeGCV(
        alphas=ALPHA_RANGE,
        fit_intercept=True,
        scale_X=False,
        scoring='pearsonr',
        store_cv_values=False,
        alpha_per_target=False,
        device=device
    )
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    
    regression.fit(x_train, y_train)
    best_alpha = float(regression.alpha_)

    ridge_model = Ridge(alpha=best_alpha)
    ridge_model.fit(x_train, y_train)
    y_predicted = ridge_model.predict(x_test)
        
    r2 = torch.stack([
            pearson_r(torch.Tensor(y_true_), torch.Tensor(y_predicted_))
            for y_true_, y_predicted_ in zip(y_test, y_predicted)
        ])
    
    r2 = r2.mean().item()

    return r2

def run_training_encoding_eval(model_args, encoding_args, encoding_eval_label, device='cuda'):

    result_dict = cp(encoding_args)
    result_dict.update(model_args)
    save_folder = os.path.join(ENCODING_EVAL_PATH, encoding_eval_label)
    os.makedirs(save_folder, exist_ok=True)

    if not check_for_duplicates(save_folder, result_dict):
        model = get_model(**model_args)
        score = calc_training_encoding_score(model=model, device=device, **encoding_args)
            
        result_dict['training_encoding_score'] = score

        label_dict = cp(encoding_args)
        label_dict['ckpt_epoch'] = model_args.get('ckpt_epoch', None)
        # Exclude None values from label
        encoding_args_str = "__".join(f"{key}_{value}" for key, value in label_dict.items() if value is not None)

        timestamp = datetime.now().strftime("%Y_%m%d_%H%M%S")
        f_save = os.path.join(save_folder, f"{timestamp}_{encoding_args_str}.json")

        with open(f_save, "w") as json_file:
            json.dump(result_dict, json_file, indent=4)
            
def calc_encoding_score_alexnet_training(subjid, roi, layerid, training_status=True, device='cuda'):

    du, ds = quick_load_preproc_images(subjid, device=device)
    x = extract_alexnet_activations(du, training_status, layerid)
    du = None
    ds = None
    
    n = compute_johnson_lindenstrauss_limit(n_samples=x.shape[0], epsilon=0.1)     
    sparse_random_projection = SparseRandomProjection(
        n_components=n,
        density=None,
        seed=0,
        allow_expansion=False
    )
    
    x = sparse_random_projection(x).cpu().numpy()      
    y, y_ = quick_load_proc_neural_data(subjid, roi)
    
    ALPHA_RANGE = [10**i for i in range(10)]    
    regression = TorchRidgeGCV(
        alphas=ALPHA_RANGE,
        fit_intercept=True,
        scale_X=False,
        scoring='pearsonr',
        store_cv_values=False,
        alpha_per_target=False,
        device=device
    )
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    
    regression.fit(x_train, y_train)
    best_alpha = float(regression.alpha_)

    ridge_model = Ridge(alpha=best_alpha)
    ridge_model.fit(x_train, y_train)
    y_predicted = ridge_model.predict(x_test)
        
    r2 = torch.stack([
            pearson_r(torch.Tensor(y_true_), torch.Tensor(y_predicted_))
            for y_true_, y_predicted_ in zip(y_test, y_predicted)
        ])
    
    r2 = r2.mean().item()
    return r2

def run_alexnet_training_encoding_eval(encoding_args, encoding_eval_label, device='cuda'):
    """
    Run encoding evaluation for AlexNet training and save results.
    encoding_args should contain:
        - subjid
        - roi
        - layerid
        - training_status: True/False
    """

    result_dict = cp(encoding_args)
    result_dict['model_type'] = 'alexnet'

    save_folder = os.path.join(ENCODING_EVAL_PATH, encoding_eval_label)
    os.makedirs(save_folder, exist_ok=True)

    if not check_for_duplicates(save_folder, result_dict):
        score = calc_encoding_score_alexnet_training(
            subjid=encoding_args['subjid'],
            roi=encoding_args['roi'],
            layerid=encoding_args['layerid'],
            training_status=encoding_args.get('training_status', True),
            device=device
        )

        result_dict['training_encoding_score'] = score

        label_dict = cp(encoding_args)
        encoding_args_str = "__".join(f"{key}_{value}" for key, value in label_dict.items())

        timestamp = datetime.now().strftime("%Y_%m%d_%H%M%S")
        f_save = os.path.join(save_folder, f"{timestamp}_{encoding_args_str}.json")

        with open(f_save, "w") as json_file:
            json.dump(result_dict, json_file, indent=4)

        print(f"Results saved to: {f_save}")
    else:
        print("Duplicate found, skipping evaluation.")