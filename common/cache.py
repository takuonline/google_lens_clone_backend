# from flask_caching import Cache
from common.search import img_search
from config import Config
import torch
from flask import g
import json


def get_yolo():
    if "yolov7_model" not in g:
        yolov7_model = torch.load(Config.YOLOV7_PATH, map_location=Config.device)
        yolov7_model.eval()
        yolov7_model.float()
        g.yolov7_model = yolov7_model

    return g.yolov7_model


def get_label_names():
    if "labels" not in g:
        with open(Config.LABELS_FILE, "r") as f:
            g.labels = json.loads(f.read())

    return g.labels


def get_embedding_model():
    if "embedding_model" not in g:
        g.embedding_model = img_search.ImgSearchModel(
            img_metadata=Config.GAME_IMGS_DATA_PATH,
            stored_img_indexes_path=Config.GAME_RESNET101_INDEX_PATH,
        )

    return g.embedding_model
