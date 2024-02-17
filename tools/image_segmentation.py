import cv2
import numpy as np
from PIL import Image
import torch
from PIL import Image
import sys
import os
import torch

sys.path.append(os.path.join(os.getcwd(), "models"))
sys.path.append(os.path.join(os.getcwd(), "models/segment_anything"))

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import sam_model_registry, sam_hq_model_registry, SamPredictor

def init_segment_models(st, image):
    predictor, model, device = initialize_ground_sam()    
    st.session_state['segmentation'] = {}
    st.session_state['segmentation']['image'] = image
    st.session_state['segmentation']['predictor'] = predictor
    st.session_state['segmentation']['model'] = model
    st.session_state['segmentation']['device'] = device

def load_image(image_pil):
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
    
def exe_ground_sam(image, text_prompt, predictor, model, device="cuda"):
    text_prompt = text_prompt
    box_threshold = 0.3
    text_threshold = 0.25

    image_pil, image = load_image(image)
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