#!/usr/bin/env python
# coding: utf-8

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import cv2
from utils import general

from PIL import Image
import traceback
from utils.common_utils import CommonUtils
from config import Config
import math


def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return img, ratio, (dw, dh)


def plot_one_box_PIL(box, img, color=None, label=None, line_thickness=None):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    line_thickness = line_thickness or max(int(min(img.size) / 200), 2)
    draw.rectangle(box, width=line_thickness, outline=tuple(color))  # plot
    if label:
        fontsize = max(round(max(img.size) / 30), 12)
        # font = ImageFont.truetype("Arial.ttf", fontsize)
        font = ImageFont.load_default()
        txt_width, txt_height = font.getsize(label)
        draw.rectangle(
            [box[0], box[1] - txt_height + 4, box[0] + txt_width, box[1]],
            fill=tuple(color),
        )
        draw.text(
            (box[0], box[1] - txt_height + 1), label, fill=(255, 255, 255), font=font
        )
    return np.asarray(img)


def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print(
            "WARNING: --img-size %g must be multiple of max stride %g, updating to %g"
            % (img_size, s, new_size)
        )
    return new_size


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def preprocessing(im, stride, imgsz):

    if im.shape[2] == 4:
        im = CommonUtils.rgba2rgb(im)

    imgsz = general.check_img_size(imgsz, s=stride)

    im = letterbox(im, imgsz, stride)[0]
    im = np.ascontiguousarray(im)

    img = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).float()
    img /= 255.0
    return img


def postprocessing(pred, img_preprocessed, im):
    pred = general.non_max_suppression(
        pred,
        conf_thres=Config.CONF_THRESHOLD,
        iou_thres=Config.IOU_THRESHOLD,
        multi_label=True,
    )

    det = pred[0]
    det[:, :4] = general.scale_coords(
        img_preprocessed.shape[2:], det[:, :4], im.shape
    ).round()

    return det


def get_img_clip(img_data, model, names: list, imgsz=640):

    im = CommonUtils.base64_2_pil(img_data.get("img_data"))

    output = dict(
        label=None,
        conf=None,
        search_img=None,
        title=None,
        bounds=[],
        im_shape=None,
        other_objects=[],
        is_crop=False,
    )

    if not im:
        # failed to convert image from base64 (base64_2_pil)
        return output

    im = np.array(im)

    stride = Config.YOLOV7_STRIDE  # model stride
    img_preprocessed = preprocessing(im, stride, imgsz)

    try:
        pred, _ = model(img_preprocessed)
    except RuntimeError:
        print("### ERROR ###")
        traceback.print_exc()
        return output

    det = postprocessing(pred, img_preprocessed, im)

    cropped_outputs = []
    largest_confidence = 0

    for *xyxy, conf, cls in det:
        # only get the cropped_output for the item with largest confidence
        if largest_confidence > conf:
            continue

        largest_confidence = conf
        bounds = [i.item() for i in xyxy]

        # convert bounds into int so that we can slice
        x1, y1, x2, y2 = [int(i.item()) for i in xyxy]
        crop_img = im[y1:y2, x1:x2]  # crop out detected object
        cropped_outputs.append(crop_img)

        label = names[int(cls)]
        title = f"{label} {conf:.2f}"

    if not cropped_outputs:
        # no object found
        crop_img, bounds = CommonUtils.crop_img(im)
        cropped_outputs.append(crop_img)
        label = None
        largest_confidence = 0
        title = None
        output["is_crop"] = True

    ## DEBUGGING ##
    # for n, i in enumerate(cropped_outputs):
    #     Image.fromarray(i).save(f"tests/clipped_output_{n}.jpg")

    pil_img = Image.fromarray(cropped_outputs[0])

    output["label"] = label
    output["conf"] = float(largest_confidence)
    output["search_img"] = pil_img
    output["title"] = title
    output["bounds"] = bounds
    output["im_shape"] = im.shape[:2]

    return output
