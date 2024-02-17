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

def crop_square(image, row, col, l):
    """
    Crop a square region from an image.

    :param image: PIL Image object.
    :param row: Row index of the center of the square.
    :param col: Column index of the center of the square.
    :param l: Edge length of the square.
    :return: Cropped PIL Image object.
    """
    width, height = image.size

    # Calculate top left corner of the square
    top_left_x = max(col - l//2, 0)
    top_left_y = max(row - l//2, 0)

    # Adjust if the square goes beyond the right or bottom edge
    top_left_x = width - l if top_left_x + l > width else top_left_x
    top_left_y = height - l if top_left_y + l > height else top_left_y

    # Adjust if the square goes beyond the left or top edge (for very small images)
    top_left_x = max(top_left_x, 0)
    top_left_y = max(top_left_y, 0)

    bottom_right_x = min(top_left_x + l, width - 1)
    bottom_right_y = min(top_left_y + l, height - 1)
    
    # Crop the image
    cropped_image = image.crop((top_left_x, top_left_y, bottom_right_x, bottom_right_y))
    
    return cropped_image, [top_left_x, top_left_y, bottom_right_x, bottom_right_y]

def pad_image(image, padding_color=0):
    width, height = image.size

    # Determine the new size (max of width and height)
    new_size = max(width, height)

    # Determine the color for padding based on the image mode
    if image.mode == "RGB":
        padding_color = (padding_color, padding_color, padding_color)
    else:
        padding_color = padding_color

    # Calculate padding for each side
    pad_width = (new_size - width) // 2
    pad_height = (new_size - height) // 2

    # Create a new image with the new size and appropriate background color
    new_img = Image.new(image.mode, (new_size, new_size), padding_color)

    # Paste the original image onto the new image, centered
    new_img.paste(image, (pad_width, pad_height))

    return new_img, [pad_width, pad_height, pad_width+width, pad_height+height]

def init_inpainting_models(st):
    model_path = 'runwayml/stable-diffusion-inpainting'
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_path, safety_checker=None)
    pipe.to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    st.session_state['inpainting'] = {}
    st.session_state['inpainting']['pipe'] = pipe
    
# def prepare_image_mask(image, mask):
#     np_mask = np.array(mask)
#     np_mask = np_mask.astype(np.float32) / 255.0
    
#     r, c = np.where(np_mask!=0)
#     min_r, max_r = np.min(r), np.max(r)
#     min_c, max_c = np.min(c), np.max(c)
    
#     np_mask[min_r:max_r, min_c:max_c] = 1.0
#     mask = Image.fromarray((np_mask * 255).astype(np.uint8))
    
#     np_image = np.array(image)
#     np_image = np_image.astype(np.float32) / 255.0
#     np_image[min_r:max_r, min_c:max_c] = 1.0
#     image = Image.fromarray((np_image * 255).astype(np.uint8))
    
#     return image, mask
    
def prepare_image_mask(image, mask, margin_ratio = 1.1, convex_hull = True):
    np_mask = np.array(mask)
    np_mask = np_mask.astype(np.float32) / 255.0
    
    r, c = np.where(np_mask!=0)
    min_r, max_r = np.min(r), np.max(r)
    min_c, max_c = np.min(c), np.max(c)
    

    np_mask[min_r:max_r, min_c:max_c] = 1.0
    mask = Image.fromarray((np_mask * 255).astype(np.uint8))
    
    # crop and pad image and mask to square
    # crop_bbox represents the region of cropped image in the original image
    mid_r, mid_c = (min_r + max_r)//2, (min_c + max_c)//2
    l = np.max((max_r - min_r + 1, max_c - min_c + 1))
    crop_l = (l * margin_ratio)//1 
    image_crop, crop_bbox = crop_square(image, mid_r, mid_c, crop_l)
    mask_crop, crop_bbox = crop_square(mask, mid_r, mid_c, crop_l)
    
    # pad image and mask to square shape
    # pad_bbox represent the region of the original image in the padded image
    image_rst, pad_bbox = pad_image(image_crop, 0)
    mask_rst, pad_bbox = pad_image(mask_crop, 0)
        
    original_size = mask_rst.size
    
    return image_rst, mask_rst, crop_bbox, pad_bbox, original_size

def recover_image(src_image, rst_image, crop_bbox, pad_bbox, original_size):
    # since the inpainting region is cropped and padded from a region in the original image
    # there is a need to paste back the rst image to the original image
    
    # step1 scale back
    rst_image = rst_image.resize(original_size)
    
    # step1 recover from padding
    tmp_image = rst_image.crop((pad_bbox[0], pad_bbox[1], pad_bbox[2], pad_bbox[3]))

    # step2 paste tmp_image to original image
    top_left_x, top_left_y = int(crop_bbox[0]), int(crop_bbox[1])
    src_image.paste(tmp_image, (top_left_x, top_left_y))
    
    return src_image

def inpainting_image(st, image, mask, background_prompt):
    cropped_image, cropped_mask, crop_bbox, pad_bbox, original_size = prepare_image_mask(image, mask)
    pipe = st.session_state['inpainting']['pipe']
    
    prompt = f"An image {background_prompt}"
    rst_image = pipe(prompt,
                     image = cropped_image, 
                     mask_image = cropped_mask, 
                     guidance = 10, 
                     strength = 1,
                     num_inference_step = 50, 
                     ).images[0]
    
    final_rst_image = recover_image(image, rst_image, crop_bbox, pad_bbox, original_size)
    
    return final_rst_image