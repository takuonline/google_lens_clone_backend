from distutils.command.config import config
from flask_restful import Resource

from common.search import img_search
from common.yolov7.get_image_clips import get_img_clip
from config.config import Config
import torch
from flask import jsonify, request


class Detect(Resource):
    # def __init__(self, *args, **kwargs) -> None:



    #     super().__init__(*args, **kwargs)

    def get(self):
        return "img_data", 200

    def post(self):

        # instantiate yolo model
        self.model = torch.load(Config.YOLOV7_PATH, map_location=Config.device)["model"]
        self.names = self.model.names  # get prediction classes
        _ = self.model.eval()
        _ = self.model.float()

        # instantiate resnet101 model
        self.embedding_model = img_search.Resnet101Search(
            stored_img_embeddings_path = Config.RESNET101_EMBEDDINGS_PATH,
            device = Config.device,
            img_metadata = Config.PNP_IMGS_DATA_PATH,

        )



        # parse request data
        img_data = request.get_json()
        num_of_results = img_data.get("num_of_results",Config.NUM_OF_PRODUCTS_RESULTS) # use default value if num_of_results is not defined


        detection_res = get_img_clip(
            img_data,
            model=self.model,
            names=self.names,
            conf_thres=0.1,
            iou_thres=0.6,
        )

        if detection_res.get("output_img",None) == None:
            return jsonify(detection_res)


        search_res = self.embedding_model.run_search(detection_res,
                                                    num_of_results=num_of_results)


        return jsonify(search_res)
