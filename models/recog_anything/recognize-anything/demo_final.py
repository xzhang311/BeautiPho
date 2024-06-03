# arguments.
#
# before you go, please download following 4 checkpoints:
# download RAM and Tag2Text checkpoints to ./pretrained/ from https://github.com/majinyu666/recognize-anything/tree/main#toolbox-checkpoints
# download GroundingDINO and SAM checkpoints to ./Grounded-Segment-Anything/ from step 1 of https://github.com/IDEA-Research/Grounded-Segment-Anything#running_man-grounded-sam-detect-and-segment-everything-with-text-prompt
import json
import time

import os
import random

import cv2
import groundingdino.datasets.transforms as T
import numpy as np
import torch
import torchvision
import torchvision.transforms as TS
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from PIL import Image, ImageDraw, ImageFont
from ram import inference_ram
from ram import inference_tag2text
from ram.models import ram_plus
from segment_anything import SamPredictor, build_sam

config_file = "../Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
ram_checkpoint = "./pretrained/ram_plus_swin_large_14m.pth"
grounded_checkpoint = "../Grounded-Segment-Anything/groundingdino_swint_ogc.pth"
sam_checkpoint = "../Grounded-Segment-Anything/sam_vit_h_4b8939.pth"
box_threshold = 0.25
text_threshold = 0.2
iou_threshold = 0.5
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

import argparse

parser = argparse.ArgumentParser(
    description='Tag2Text inferece for tagging and captioning')
parser.add_argument('--image',
                    metavar='DIR',
                    help='path to dataset',
                    default='images/demo/demo1.jpg')


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(
        clean_state_dict(checkpoint["model"]), strict=False)
    #print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
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
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(
            logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases



@torch.no_grad()
def inference(img_name,
    raw_image, specified_tags, do_det_seg,
    tagging_model_type, tagging_model, grounding_dino_model, sam_model
):
    raw_image = raw_image.convert("RGB")

    # run tagging model
    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = TS.Compose([
        TS.Resize((384, 384)),
        TS.ToTensor(),
        normalize
    ])

    image = raw_image.resize((384, 384))
    image = transform(image).unsqueeze(0).to(device)

    res = inference_ram(image, tagging_model)
    tags = res[0].strip(' ').replace('  ', ' ').replace(' |', ',')
    tags_chinese = res[1].strip(' ').replace('  ', ' ').replace(' |', ',')

    # run groundingDINO
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image, _ = transform(raw_image, None)  # 3, h, w

    boxes_filt, scores, pred_phrases = get_grounding_output(
        grounding_dino_model, image, tags, box_threshold, text_threshold, device=device
    )

    # run SAM
    image = np.asarray(raw_image)
    sam_model.set_image(image)

    size = raw_image.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    # use NMS to handle overlapped boxes
    nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]

    with open('%s.json'%img_name, 'w') as f:
        data_final = {'bboxes':boxes_filt.numpy().astype('int').tolist(), 'labels':pred_phrases}
        json.dump(data_final, f)


if __name__ == "__main__":

    args = parser.parse_args()

    # load RAM
    ram_model = ram_plus(pretrained=ram_checkpoint, image_size=384, vit='swin_l')
    ram_model.eval()
    ram_model = ram_model.to(device)

    # load Tag2Text
    delete_tag_index = []  # filter out attributes and action categories which are difficult to grounding
    for i in range(3012, 3429):
        delete_tag_index.append(i)

    # load groundingDINO
    grounding_dino_model = load_model(config_file, grounded_checkpoint, device=device)

    # load SAM
    sam_model = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))


    def inference_with_ram(img_name, img, do_det_seg):
        return inference(img_name, img, None, do_det_seg, "RAM", ram_model, grounding_dino_model, sam_model)


    img_file = args.image
    img_name = img_file.split('/')[-1].split('.')[0]
    in_img = Image.open(img_file)
    in_img = in_img.resize((512, 512))

    inference_with_ram(img_name, in_img, do_det_seg=True)