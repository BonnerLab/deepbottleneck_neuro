import os
import pandas as pd
import xarray as xr
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from main import *
from torch_cv import *
from sklearn.linear_model import Ridge
from regression import *
from helpers.initialise_models import * 
from helpers.encoding_score_tools_new import * 
from bonner.models.hooks import *
# from calc_encoding import get_model
from copy import deepcopy as cp
from datetime import datetime
import json
from sklearn.model_selection import train_test_split

# File paths
MAJAJ_IMAGES = '/data/shared/brainscore/brainio/image_dicarlo_hvm-public'
MAJAJ_DATA = '/data/shared/datasets/majajhong'
MAJAJ_NAME_DICT = '/data/shared/brainscore/image_dicarlo_hvm-public.csv'
ENCODING_EVAL_PATH = "/data/apassi1/deepbottleneck_data/encoding_eval"

def get_images_and_neural_data_for_monkey(monkey_name, roi):

    file_path = f'/data/shared/datasets/majajhong/SUBJECT_{monkey_name}_REGION_{roi}'
    neural_data = xr.open_dataset(file_path, engine='netcdf4')
    y = neural_data['x'].values
    y = np.squeeze(y)

    images = [os.path.join(MAJAJ_IMAGES, filename) for filename in os.listdir(MAJAJ_IMAGES)]
    name_dict = pd.read_csv(MAJAJ_NAME_DICT).set_index('image_file_name')['image_id'].to_dict()
    i = neural_data['x'].stimulus_id.values
    
    inverse_name_dict = {v: k for k, v in name_dict.items()}
    ordered_image_files_base_names = [inverse_name_dict[image_id] for image_id in i]
    ordered_image_files = [os.path.join(MAJAJ_IMAGES, base_name) for base_name in ordered_image_files_base_names]
    image_tensors = []
    transform = transforms.ToTensor()

    # Load images and convert to tensors
    for img_path in ordered_image_files:
        img = Image.open(img_path)
        img_tensor = transform(img)
        image_tensors.append(img_tensor)

    images = torch.stack(image_tensors, dim=0)
    images = preprocess(images)

    return images, y

def extract_all_activations(images, model):
    batch_size = 50
    num_samples = len(images)
    model = model.cuda()
    x = []
    
    for i in range(0, num_samples, batch_size):
        batch_input = images[i:i + batch_size]
        batch_input = batch_input.cuda()  # Move the batch to GPU
        
        with torch.no_grad():
            output = model(batch_input)
        output = output.full_view()
        output = output.reshape(output.shape[0], -1)
        output = output.cpu()
        x.append(output)
    
    x = torch.cat(x, dim=0)
    return x

def regression_random_split(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    model: Ridge = Ridge(),
    test_size: float = 0.2,
    random_state: int = 42,
):

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    # Fit the model
    model.fit(x_train, y_train)

    # Predict on the test set
    y_predicted = model.predict(x_test)

    return torch.Tensor(y_test), torch.Tensor(y_predicted)

def monkey_regression(x, y, random_state: int = 42):
    
    ALPHA_RANGE = [10**i for i in range(10)]
    regression = TorchRidgeGCV(
        alphas=ALPHA_RANGE,
        fit_intercept=True,
        scale_X=False,
        scoring='pearsonr',
        store_cv_values=False,
        alpha_per_target=False,
        device='cuda'
    )

    regression.fit(x, y)
    best_alpha = float(regression.alpha_)
    
    y_true, y_predicted = regression_random_split(
        x=torch.Tensor(x), y=torch.Tensor(y), model=Ridge(alpha=best_alpha), random_state= random_state)
    
    y_true = y_true.T
    y_predicted = y_predicted.T
    
    r2 = torch.stack([
        pearson_r(y_true_, y_predicted_)
        for y_true_, y_predicted_ in zip(y_true, y_predicted)
    ])
    return r2

def calc_encoding_score_monkey(model,roi,layerid,device='cuda'):
    
    images1, y1 = get_images_and_neural_data_for_monkey("Chabo", roi)
    images2, y2 = get_images_and_neural_data_for_monkey("Tito", roi)
    training_status = True
    
    if model == "AlexNet": 
        x1 = extract_alexnet_activations(images1, training_status, layerid)
        x2 = extract_alexnet_activations(images2, training_status, layerid)
    else:
        model = model[:(layerid+1)]
        x1 = extract_all_activations(images1, model)
        x2 = extract_all_activations(images2, model)

    e1 = monkey_regression(x1, y1)
    e2 = monkey_regression(x2, y2)
    e = torch.cat((e1, e2), dim=0)
    
    return e.mean()

def run_encoding_eval_monkey(model_args,encoding_args,encoding_eval_label,device='cuda'):
    print("Hello!")
    result_dict = cp(encoding_args)
    result_dict.update(model_args)
    save_folder = os.path.join(ENCODING_EVAL_PATH, encoding_eval_label)
    os.makedirs(save_folder, exist_ok=True)

    if model_args[model_label] != "AlexNet": 
        model = get_model(**model_args)
    e = calc_encoding_score_monkey(model, device='cuda', **encoding_args)
    result_dict['encoding_score']=e.numpy().item()
    
    label_dict = cp(encoding_args)
    label_dict['ckpt_epoch']=model_args['ckpt_epoch']
    encoding_args_str = "__".join(f"{key}_{value}" for key, value in label_dict.items())
    
    timestamp = datetime.now().strftime("%Y_%m%d_%H%M%S")
    f_save=os.path.join(save_folder,f"{timestamp}_{encoding_args_str}.json")
        
    with open(f_save, "w") as json_file:
        json.dump(result_dict, json_file, indent=4)

