from flask_restful import Resource

from common.search import img_search
from common.yolov7.get_image_clips import get_img_clip
from config import Config
import torch
from flask import jsonify, request
from common.cache import get_embedding_model


class ImgSearch(Resource):
    def post(self):

        # instantiate resnet101 model
        embedding_model = get_embedding_model()

        # parse request data
        img_data = request.get_json()
        num_of_results = img_data.get(
            "num_of_results", Config.NUM_OF_PRODUCTS_RESULTS
        )  # use default value if num_of_results is not defined

        # search the image clip in product database
        search_res = embedding_model.search(img_data, num_of_results=num_of_results)

        return jsonify(search_res)
