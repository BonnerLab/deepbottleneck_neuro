# from score_models import run_encoding_eval
import sys
import os
import re

sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from calc_encoding import run_encoding_eval

models = ["new_imagenet-1k_rand",
          "new_imagenet-1k_pca"]

models = ["new_imagenet-1k_pca"]

# Original list of checkpoint IDs
layerids = [1, 3, 5, 7, 9, 11]
ckpts = [0, 2400]
subjids = [0, 1, 2, 3, 4, 5, 6, 7]

encoding_eval_label = "new_imagenet-1k"

for model in models:
    for ckpt in ckpts: 
        
        model_args=dict(model_type='trained_bottleneck',
                    model_label = model, # name of folder from .json
                    ckpt_epoch = ckpt)

        for subjid in subjids:  
            for layerid in layerids: 
                encoding_args=dict(subjid=subjid, roi='ventral visual stream', layerid=layerid)
                run_encoding_eval(model_args,encoding_args,encoding_eval_label,device='cuda:0')
