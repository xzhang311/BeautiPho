import time
import numpy as np
import streamlit as st
from PIL import Image
from utils import initialize_ground_sam, exe_ground_sam

st.set_page_config(layout="wide")

def ui():
    # Main
    st.text("")
    with st.container():
        st.text(" ")
        st.subheader("Implementation")
        st.write("From the sidebar import image and select any method,algorithm to perform segmentation on your image")

    # Sidebar

    # Model import
    with st.container():
        st.sidebar.header("Import image: ")
        file = st.sidebar.file_uploader("Upload your file:", ['png', 'jpg', 'jpeg'], accept_multiple_files=False)

        if 'imported' not in st.session_state:
            if st.sidebar.button("Import"):
                st.session_state['imported'] = True
                with st.spinner("Importing image...."):
                    time.sleep(1)
                st.sidebar.success("Imported successfully")
                st.sidebar.image(file)
        else:
            st.sidebar.image(file)

    # Search for
    with st.container():
        st.sidebar.header("Search for: ")
        prompt = st.sidebar.text_input('Provide prompt', 'Human')

    # Model
    with st.container():
        if st.sidebar.button("OK"):
            if file is None:
                st.sidebar.error("Please upload an image first.")
            else:
                with st.spinner("Performing Segmentation...."):
                    mask = exe_ground_sam(file, prompt, st.session_state['predictor'], st.session_state['model'])
                    st.image(mask)
                st.sidebar.success("Done ! ")

def init_models():
    if 'predictor' not in st.session_state or \
        'model' not in st.session_state:
        predictor, model, device = initialize_ground_sam()    
        st.session_state['predictor'] = predictor
        st.session_state['model'] = model
        st.session_state['device'] = device

if __name__ == "__main__":
    init_models();
    ui()