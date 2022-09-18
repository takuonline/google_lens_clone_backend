from flask_restful import Resource
from common.yolov7.get_image_clips import get_img_clip
from config import Config
from flask import jsonify, request
from common import cache

class Detect(Resource):

    def get(self):
        return "img_data", 200

    def post(self):
        embedding_model = cache.get_embedding_model()
        yolo_model = cache.get_yolo()
        names = cache.get_label_names()  # get prediction classes

        # parse request data
        img_data = request.get_json()
        num_of_results = img_data.get(
            "num_of_results", Config.NUM_OF_PRODUCTS_RESULTS
        )  # use default Config.NUM_OF_PRODUCTS_RESULTS value if num_of_results is not defined

        # find object in image
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
