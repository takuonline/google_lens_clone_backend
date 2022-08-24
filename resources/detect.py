from flask_restful import Resource

from common.search import img_search
from common.yolov7.get_image_clips import get_img_clip
from config import Config
import torch
from flask import jsonify, request


class Detect(Resource):
    # def __init__(self, *args, **kwargs) -> None:

    #     super().__init__(*args, **kwargs)

    def get(self):
        return "img_data", 200

    def post(self):

        # instantiate yolo model
        model = torch.load(Config.YOLOV7_PATH, map_location=Config.device)["model"]
        names = model.names  # get prediction classes
        _ = model.eval()
        _ = model.float()

        # # instantiate resnet101 model
        # embedding_model = img_search.Resnet101Search(
        #     img_metadata = Config.PNP_IMGS_DATA_PATH,
        #     stored_img_embeddings_path = Config.PNP_RESNET101_EMBEDDINGS_PATH,
        #     device = Config.device,
        # )
        # instantiate resnet101 model
        embedding_model = img_search.ImgSearchModel(
            img_metadata=Config.GAME_IMGS_DATA_PATH,
            stored_img_indexes_path=Config.GAME_RESNET101_INDEX_PATH,
            device=Config.device,
        )

        # parse request data
        img_data = request.get_json()
        num_of_results = img_data.get(
            "num_of_results", Config.NUM_OF_PRODUCTS_RESULTS
        )  # use default value if num_of_results is not defined

        detection_res = get_img_clip(
            img_data,
            model=model,
            names=names,
        )

        if detection_res.get("search_img", None) == None:
            return jsonify(detection_res)

        # search the image clip in out database
        search_res = embedding_model.search(
            detection_res, num_of_results=num_of_results
        )

        return jsonify(search_res)
