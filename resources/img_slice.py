from flask_restful import Resource
import base64
import io
from PIL import Image
from common.yolov7.get_image_clips import get_img_clip
from config.config import Config
import torch
from pathlib import Path
from flask import Flask, jsonify, request


def base64_2_pil(data):
    decoded = base64.b64decode(data)
    img = Image.open(io.BytesIO(decoded))
    return img

def pil2base64(im):
    buffered = io.BytesIO()
    im.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

class Detect(Resource):
    def __init__(self, *args, **kwargs) -> None:

        # instantiate yolo model
        self.model = torch.load(Config.MODEL_W6_DIR, map_location=Config.device)[
            "model"
        ]
        self.names = self.model.names
        _ = self.model.eval()
        _ = self.model.float()

        super().__init__(*args, **kwargs)

    def get(self):
        return "img_data", 200

    def post(self):
        img_data = request.get_json()
        img = base64_2_pil(img_data.get("img_data"))
        # img.save("output.jpeg")
        label, output_img = get_img_clip(
            img,
            model=self.model,
            names=self.names,
            conf_thres=0.2,
            iou_thres=0.6,
        )
        res = dict()
        res["label"] = label
        res["output_img"] = pil2base64(Image.fromarray(output_img))

        return jsonify(res)


