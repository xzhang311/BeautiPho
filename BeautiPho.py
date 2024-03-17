import time
import streamlit as st
from PIL import Image
from tools.utils import (del_model, release_cache)
from tools.image_segmentation import (init_segment_models, exe_ground_sam)
from tools.image_captioning import (init_caption_models, get_background_description)
from tools.image_inpainting_sd import (init_inpainting_models_sd, inpainting_image_sd)
from tools.image_inpainting_lama import (init_inpainting_models_lama, inpainting_image_lama)
from tools.face_replacing import (init_face_replacing_models, exe_insightface)

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
        inpainting_model_selection = st.sidebar.selectbox("Inpainting model", ("SD", "LAMA"))
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
                    
                    if inpainting_model_selection == 'SD':
                        init_inpainting_models_sd(st)
                        image_inpaint = inpainting_image_sd(st, st.session_state['segmentation']['image'],
                                        st.session_state['segmentation']['mask'],
                                        background_prompt)
                    else:
                        init_inpainting_models_lama(st)
                        image_inpaint = inpainting_image_lama(st, st.session_state['segmentation']['image'],
                                        st.session_state['segmentation']['mask'])
                        
                    st.image(image_inpaint)
                    st.session_state['inpainting_done'] = True
                    
                    if inpainting_model_selection == 'SD':
                        del st.session_state['inpainting']['pipe']
                    else:
                        del st.session_state['inpainting']['model']
                        
                    release_cache()
                    st.session_state['inpainting_done'] = True
                st.sidebar.success("Done ! ")

    # Face Refinement
    with st.container():
        st.sidebar.header("Face Refinement: ")

    with st.container():
        face_image_path = st.sidebar.file_uploader("Upload a face image:", ['png', 'jpg', 'jpeg'], accept_multiple_files=False)
        if face_image_path is not None:
            face_image = Image.open(face_image_path).convert("RGB")

        if 'face_imported' not in st.session_state:
            if st.sidebar.button("Import Face Image"):
                st.session_state['face_imported'] = True
                with st.spinner("Importing face image...."):
                    time.sleep(1)
                st.sidebar.success("Face image imported successfully")
                st.sidebar.image(face_image)
        else:
            st.sidebar.image(face_image)


    # Assuming a function to execute face refinement
    with st.container():
        # face_refinement_action = st.sidebar.selectbox("Face Refinement Action:",
        #                                               ["Choose...", "Replace Face", "Enhance Face Features"])
        # if face_refinement_action != "Choose...":
        #     if 'face_refinement_imported' not in st.session_state or not st.session_state['face_refinement_imported']:
        #         st.sidebar.error("Please upload a face image first.")
        #     else:

        if st.sidebar.button("Face Replacing"):
            if face_image_path is None:
                st.sidebar.error("Please upload a face image first.")
            else:
                with st.spinner(f"Performing Face Replacing...."):
                    init_face_replacing_models(st, image, face_image)
                    refined = exe_insightface(image, face_image,
                                              st.session_state['insightface']['app'],
                                              st.session_state['insightface']['swapper'])
                    st.image(refined, caption=f"Image after Face Replacing")
                    # Delete the model object.
                    del (st.session_state['insightface']['app'])
                    del (st.session_state['insightface']['swapper'])
                    release_cache()

                    time.sleep(2)  # Simulate processing time
                    st.session_state['face_replacing_done'] = True
                st.sidebar.success("Done ! ")


def init_status():
    if 'segmentation_done' not in st.session_state:
        st.session_state['segmentation_done'] = False

    if 'inpainting_done' not in st.session_state:
        st.session_state['inpainting_done'] = False

    if 'face_replacing_done' not in st.session_state:
        st.session_state['face_replacing_done'] = False


if __name__ == "__main__":
    ui()
    