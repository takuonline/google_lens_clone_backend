from flask_restful import Resource

from common.search import img_search
from common.yolov7.get_image_clips import get_img_clip
from config import Config
import torch
from flask import jsonify, request

from common.cache import get_yolo, get_embedding_model


class Detect(Resource):
    # def __init__(self, *args, **kwargs) -> None:

    #     super().__init__(*args, **kwargs)

    def get(self):
        return "img_data", 200

    def post(self):
        embedding_model = get_embedding_model()
        yolo_model = get_yolo()

        names = yolo_model.names  # get prediction classes

        # parse request data
        img_data = request.get_json()
        num_of_results = img_data.get(
            "num_of_results", Config.NUM_OF_PRODUCTS_RESULTS
        )  # use default Config.NUM_OF_PRODUCTS_RESULTS value if num_of_results is not defined

        detection_res = get_img_clip(
            img_data,
            model=yolo_model,
            names=names,
        )

        # search the image clip in out database
        search_res = embedding_model.search_clip(
            detection_res, num_of_results=num_of_results
        )

        return jsonify(search_res)
