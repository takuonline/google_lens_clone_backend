from pathlib import Path
import torch


class Config:

    BASE_PATH = Path().absolute() / Path("common") / Path("ml_models")
    MODEL_E6_DIR = BASE_PATH / Path("yolov7-e6.pt")
    MODEL_DIR = BASE_PATH / Path("yolov7.pt")
    MODEL_W6_DIR = BASE_PATH / Path("yolov7-w6.pt")
    MODEL_X_DIR = BASE_PATH / Path("yolov7x.pt")

    device = torch.device("cpu")
