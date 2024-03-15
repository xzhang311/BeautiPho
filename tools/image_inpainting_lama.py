import logging
import os
import sys
import traceback

import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
import os
import sys
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
from PIL import Image

sys.path.append(os.path.join(os.getcwd(), "models"))
sys.path.append(os.path.join(os.getcwd(), "models/lama"))

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

def init_inpainting_models_lama(st):    
    with open('models/lama/configs/prediction/default.yaml', 'r') as file:
        dict_config = yaml.safe_load(file)
    predict_config = OmegaConf.create(dict_config) 
    predict_config.model.path = 'model_weights/big-lama'
    
    device = torch.device('cuda')
    train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
        
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    out_ext = predict_config.get('out_ext', '.png')

    checkpoint_path = os.path.join(predict_config.model.path, 
                                    'models', 
                                    predict_config.model.checkpoint)    
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    if not predict_config.get('refine', False):
        model.to(device)
        
    st.session_state['inpainting'] = {}
    st.session_state['inpainting']['model'] = model
    st.session_state['inpainting_config_lama'] = predict_config

def inpainting_image_lama(st, image, mask):
    predict_config = st.session_state['inpainting_config_lama']
    model = st.session_state['inpainting']['model']
    
    image = torch.tensor(np.array(image)/255)
    image = image.permute(2, 0, 1)
    image = image.unsqueeze(0)
    image = image.to(predict_config.device).float()
    mask = torch.tensor((np.array(mask)/255 > 0) * 1)
    mask = mask.unsqueeze(0)
    mask = mask.unsqueeze(0)
    mask = mask.to(predict_config.device).float()
    batch = {'image': image, 'mask': mask}
    
    if predict_config.get('refine', False):
        assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
        # image unpadding is taken care of in the refiner, so that output image
        # is same size as the input image
        cur_res = refine_predict(batch, model, **predict_config.refiner)
        cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy()
    else:  
        batch = model(batch)                    
        cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
        unpad_to_size = batch.get('unpad_to_size', None)
        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]
            
    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    final_rst_image = Image.fromarray(cur_res)
    
    return final_rst_image