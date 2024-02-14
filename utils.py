import cv2
import numpy as np
from PIL import Image
import streamlit as st
import json
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import itertools
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "models"))
sys.path.append(os.path.join(os.getcwd(), "models/segment_anything"))

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import sam_model_registry, sam_hq_model_registry, SamPredictor

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def dilate(mask, times = 1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    for i in range(times):
        mask = cv2.dilate(mask, kernel)
    return mask

def initialize_ground_sam():
    config_file = 'models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
    grounded_checkpoint = 'model_weights/groundingdino_swint_ogc.pth'
    sam_version = 'vit_h'
    sam_checkpoint = 'model_weights/sam_vit_h_4b8939.pth'
    box_threshold = 0.3
    text_threshold = 0.25
    device = "cuda"
    use_sam_hq = False
    
    predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    model = load_model(config_file, grounded_checkpoint, device=device)
    
    return predictor, model, device
    
def exe_ground_sam(image_path, text_prompt, predictor, model, device="cuda"):
    text_prompt = text_prompt
    box_threshold = 0.3
    text_threshold = 0.25

    image_pil, image = load_image(image_path)
    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, device=device
    )

    image = np.array(image_pil)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    
    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )
    
    mask_img = torch.zeros(masks.shape[-2:])
    mask = masks[0]
    mask_img[mask.cpu().numpy()[0] == True] = 255
    mask_img = dilate(mask_img.numpy()/255, times = 3) * 255
    mask_img = Image.fromarray(mask_img)
    mask_img = mask_img.convert("L")
    
    return mask_img
    
def instance_seg(img_file, search, thresh):
    path_to_model = 'model_weights/frozen_inference_graph_coco.pb'
    path_to_config = 'model_weights/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'

    model = cv2.dnn.readNetFromTensorflow(path_to_model, path_to_config)

    colors = np.random.randint(125, 255, (80, 3))

    # Choices
    segment_class_id = []

    if search == 'Person':
        segment_class_id = [0]
    elif search == 'Motorcycle':
        segment_class_id = [1]
    elif search == 'Car':
        segment_class_id = [2]
    elif search == 'All':
        segment_class_id = []

    if img_file is None:
        st.write("Error: could not read image")
        exit()
    else:
        img = np.asarray(img_file)
        img = cv2.resize(img, (650, 550), interpolation=cv2.INTER_LINEAR)
        height, width, _ = img.shape

        black_image = np.zeros((height, width, 3), np.uint8)
        black_image[:] = (0, 0, 0)
        blob = cv2.dnn.blobFromImage(img, swapRB=True)
        model.setInput(blob)
        boxes, masks = model.forward(["detection_out_final", "detection_masks"])
        detection_count = boxes.shape[2]

        count = 0
        for i in range(detection_count):
            box = boxes[0, 0, i]
            class_id = box[1]
            score = box[2]

            if score < thresh or (segment_class_id and class_id not in segment_class_id):
                continue
            x = int(box[3] * width)
            y = int(box[4] * height)
            x2 = int(box[5] * width)
            y2 = int(box[6] * height)

            roi = black_image[y: y2, x: x2]
            roi_height, roi_width, _ = roi.shape
            mask = masks[i, int(class_id)]
            mask = cv2.resize(mask, (roi_width, roi_height))
            _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
            cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 3)
            contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = colors[int(class_id)]
            for cnt in contours:
                cv2.fillPoly(roi, [cnt], (int(color[0]), int(color[1]), int(color[2])))
            count += 1
            # cv2.imshow("Final", np.hstack([img, black_image]))
            # cv2.imshow("Overlay_image", ((0.6 * black_image) + (0.4 * img)).astype("uint8"))
        if count == 0:
            st.write("No objects found.")
        else:
            st.image(np.hstack([img, black_image]), channels="BGR", caption="Instance Segmentation")
            st.text(f"No of segmented objects : {count}")
            cv2.waitKey(0)


def semantic_seg(img, search, thresh):
    global segment_class_id
    import cv2
    import numpy as np

    # Load the model and configuration files
    path_to_model = 'frozen_inference_graph_coco.pb'
    path_to_config = 'mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'

    model = cv2.dnn.readNetFromTensorflow(path_to_model, path_to_config)

    segment_class_id = []
    # Choose the class to segment
    if search == 'Person':
        segment_class_id = [0]
    elif search == 'Motorcycle':
        segment_class_id = [1]
    elif search == 'Car':
        segment_class_id = [2]

    # Load the input image
    img_file = img
    img_file = np.asarray(img_file)
    # Resize the image for faster processing
    img_file = cv2.resize(img_file, (650, 550), interpolation=cv2.INTER_LINEAR)
    height, width, _ = img_file.shape

    # Create a black image with the same size as the input image
    black_image = np.zeros((height, width, 3), np.uint8)
    black_image[:] = (0, 0, 0)

    # Preprocess the input image
    blob = cv2.dnn.blobFromImage(img_file, swapRB=True)
    model.setInput(blob)

    # Forward pass through the model
    boxes, masks = model.forward(["detection_out_final", "detection_masks"])
    detection_count = boxes.shape[2]

    # Iterate over the detected objects
    for i in range(detection_count):
        box = boxes[0, 0, i]
        class_id = box[1]
        score = box[2]

        # Filter out the objects that do not match the chosen class or have a low score
        if score < thresh or class_id != segment_class_id:
            continue

        # Get the bounding box coordinates and extract the ROI from the black image
        x = int(box[3] * width)
        y = int(box[4] * height)
        x2 = int(box[5] * width)
        y2 = int(box[6] * height)
        roi = black_image[y:y2, x:x2]

        # Resize the mask to match the size of the ROI
        mask = masks[i, int(class_id)]
        mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]))

        # Threshold the mask and apply the bitwise AND operation to extract the segmented object
        _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)
        img = cv2.bitwise_and(img_file[y:y2, x:x2], img_file[y:y2, x:x2], mask=mask)

        # Fill the segmented object with a random color
        color = np.random.randint(0, 255, (3,))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.fillPoly(roi, [cnt], (int(color[0]), int(color[1]), int(color[2])))

    # Display the segmented image
    st.image(np.hstack([img_file, black_image]), channels="BGR", caption="Semantic Segmentation")
    cv2.waitKey(0)