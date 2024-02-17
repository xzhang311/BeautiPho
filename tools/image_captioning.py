import requests
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
# from .utils import del_model, release_cache
from pathlib import Path
import streamlit as st

def init_caption_models(st, image, mask):
    # processor and model initialization
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    st.session_state['caption'] = {}
    st.session_state['caption']['image'] = image
    st.session_state['caption']['mask'] = mask
    st.session_state['caption']['processor'] = processor
    st.session_state['caption']['model'] = model
    
def caption_image(st, prompt = None):
    pil_image = st.session_state['caption']['image']
    
    # processor and model initialization
    processor = st.session_state['caption']['processor']
    model = st.session_state['caption']['model']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if prompt is None:
        inputs = processor(pil_image, return_tensors="pt").to(device, torch.float16)

        generated_ids = model.generate(**inputs, max_new_tokens=20)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(generated_text)
    else:
        inputs = processor(pil_image, text=prompt, return_tensors="pt").to(device, torch.float16)

        generated_ids = model.generate(**inputs, max_new_tokens=20)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(generated_text)
        
    # del_model(processor)
    # del_model(model)
    # release_cache()
        
    return generated_text

def get_background_description(st):
    pil_image = st.session_state['segmentation']['image']
    prompt = "Question: Where was this photo taken. Answer:"
    caption = caption_image(st, prompt=prompt)
    return caption
    
if __name__ == "__main__":
    image_path = Path("/mnt/ebs_xizhn_proj/BeautiPho/data/71AsIBYB1pL (2).jpg")
    pil_image = Image.open(image_path)
    
    caption_image(pil_image)
    prompt = "Question: how many legs does this chair have? Answer:"
    caption_image(pil_image, prompt=prompt)