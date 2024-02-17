from diffusers import StableDiffusionInpaintPipeline
import numpy as np
import PIL
import torch
from PIL import Image
import numpy as np
from pathlib import Path
from diffusers import DDIMScheduler, LMSDiscreteScheduler, DPMSolverMultistepScheduler
import argparse
import os
import clip
import cv2
from scipy.ndimage import binary_dilation

def init_inpainting_models(st):
    model_path = 'runwayml/stable-diffusion-inpainting'
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_path, safety_checker=None)
    pipe.to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    st.session_state['inpainting'] = {}
    st.session_state['inpainting']['pipe'] = pipe
    
def prepare_image_mask(image, mask):
    np_mask = np.array(mask)
    np_mask = np_mask.astype(np.float32) / 255.0
    
    r, c = np.where(np_mask!=0)
    min_r, max_r = np.min(r), np.max(r)
    min_c, max_c = np.min(c), np.max(c)
    
    np_mask[min_r:max_r, min_c:max_c] = 1.0
    mask = Image.fromarray((np_mask * 255).astype(np.uint8))
    
    np_image = np.array(image)
    np_image = np_image.astype(np.float32) / 255.0
    np_image[min_r:max_r, min_c:max_c] = 1.0
    image = Image.fromarray((np_image * 255).astype(np.uint8))
    
    return image, mask
    
def inpainting_image(st, image, mask, background_prompt):
    image, mask = prepare_image_mask(image, mask)
    pipe = st.session_state['inpainting']['pipe']
    
    prompt = f"An image {background_prompt}"
    rst_image = pipe(prompt,
                     image = image, 
                     mask_image = mask, 
                     guidance = 10, 
                     strength = 1,
                     num_inference_step = 50, 
                     ).images[0]
    
    return rst_image