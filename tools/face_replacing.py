import cv2
import numpy as np
from PIL import Image
import torch
from PIL import Image
import sys
import os
import torch

sys.path.append(os.path.join(os.getcwd(), "models"))
sys.path.append(os.path.join(os.getcwd(), "models/InsightFace"))

import InsightFace.python_package.insightface as insightface
from InsightFace.python_package.insightface.app import FaceAnalysis
from InsightFace.python_package.insightface.data import get_image as ins_get_image


def init_face_replacing_models(st, image, face):
    app, swapper, device = initialize_insightface()
    st.session_state['insightface'] = {}
    st.session_state['insightface']['image'] = image
    st.session_state['insightface']['face'] = face
    st.session_state['insightface']['app'] = app
    st.session_state['insightface']['swapper'] = swapper
    st.session_state['insightface']['device'] = device


def initialize_insightface():
    device = "cuda"

    md_path = 'model_weights'
    app = FaceAnalysis(name='buffalo_l', root=md_path)
    app.prepare(ctx_id=0, det_size=(640, 640))

    swapper = insightface.model_zoo.get_model('model_weights/insightface/inswapper_128.onnx')
    
    return app, swapper, device


def exe_insightface(image_pil, face_pil, app, swapper, device="cuda"):
    images = ins_get_image(image_pil)
    image = app.get(images)
    image = sorted(image, key = lambda x : x.bbox[0])

    face = ins_get_image(face_pil)
    face = app.get(face)
    face = sorted(face, key = lambda x : x.bbox[0])
    face_src = face[0]

    for img_face in image:
        images = swapper.get(images, img_face, face_src, paste_back=True)

    images = images[:, :, ::-1]
    images = Image.fromarray(images)
    
    return images