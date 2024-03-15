import time
import streamlit as st
from PIL import Image
from tools.utils import (del_model, release_cache)
from tools.image_segmentation import (init_segment_models, exe_ground_sam)
from tools.image_captioning import (init_caption_models, get_background_description)
from tools.image_inpainting import (init_inpainting_models, inpainting_image)
st.set_page_config(layout="wide")

def ui():
    # Main
    st.text("")
    with st.container():
        st.text(" ")
        st.subheader("Implementation")
        st.write("From the sidebar import image and select any method,algorithm to perform segmentation on your image")

    # Model import
    with st.container():
        st.sidebar.header("Import image: ")
        image_path = st.sidebar.file_uploader("Upload your file:", ['png', 'jpg', 'jpeg'], accept_multiple_files=False)
        if image_path is not None:
            image = Image.open(image_path).convert("RGB")
        
        if 'imported' not in st.session_state:
            if st.sidebar.button("Import"):
                st.session_state['imported'] = True
                with st.spinner("Importing image...."):
                    time.sleep(1)
                st.sidebar.success("Imported successfully")
                st.sidebar.image(image)
        else:
            st.sidebar.image(image)

    # Segment for
    with st.container():
        st.sidebar.header("Segment for: ")
        prompt = st.sidebar.text_input('Provide prompt', 'Human')

    # Segmentation Model
    with st.container():
        if st.sidebar.button("Segment target"):
            if image_path is None:
                st.sidebar.error("Please upload an image first.")
            else:
                with st.spinner("Performing Segmentation...."):
                    init_segment_models(st, image)
                    mask = exe_ground_sam(image, prompt, st.session_state['segmentation']['predictor'], st.session_state['segmentation']['model'])
                    if 'mask' not in st.session_state['segmentation']:
                        st.session_state['segmentation']['mask'] = mask
                    st.image(mask)
                    # Delete the model object.
                    del(st.session_state['segmentation']['predictor'])
                    del(st.session_state['segmentation']['model'])
                    release_cache()
                    st.session_state['segmentation_done'] = True
                st.sidebar.success("Done ! ")

    # Inpaint background
    with st.container():
        if st.sidebar.button("Inpaint background"):
            if st.session_state['segmentation_done'] != True:
                st.sidebar.error("Please segment the image first.")
            else:
                with st.spinner("Performing Inpainting...."):
                    init_caption_models(st, 
                                        st.session_state['segmentation']['image'],
                                        st.session_state['segmentation']['mask'])
                    background_prompt = get_background_description(st)
                    st.text(background_prompt)
                    del(st.session_state['caption'])
                    release_cache()
                    init_inpainting_models(st)
                    image_inpaint = inpainting_image(st, st.session_state['segmentation']['image'],
                                     st.session_state['segmentation']['mask'],
                                     background_prompt)
                    st.image(image_inpaint)
                    st.session_state['inpainting_done'] = True
                    del st.session_state['inpainting']['pipe']
                    release_cache()
                    st.session_state['inpainting_done'] = True
                st.sidebar.success("Done ! ")

def init_status():
    if 'segmentation_done' not in st.session_state:
        st.session_state['segmentation_done'] = False

    if 'inpainting_done' not in st.session_state:
        st.session_state['inpainting_done'] = False
        

if __name__ == "__main__":
    ui()
    