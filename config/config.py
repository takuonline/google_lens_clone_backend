from pathlib import Path
import torch


class Config:

    BASE_PATH = Path().absolute()
    ML_MODEL_DIR = BASE_PATH / Path("common") / Path("ml_models")
    DATA_DIR = BASE_PATH / Path("data")

    YOLOV7_PATH = ML_MODEL_DIR / Path("yolov7.pt")
    YOLOV7_W6_PATH = ML_MODEL_DIR / Path("yolov7-w6.pt")
    YOLOV7_X_PATH = ML_MODEL_DIR / Path("yolov7x.pt")
    YOLOV7_E6_PATH = ML_MODEL_DIR / Path("yolov7-e6.pt")
    YOLOV7_D6_PATH = ML_MODEL_DIR / Path("yolov7-d6.pt")
    YOLOV7_E6E_PATH = ML_MODEL_DIR / Path("yolov7-e6e.pt")


    device = torch.device("cpu")

    RESNET101_EMBEDDINGS_PATH = DATA_DIR / Path("embeddings") / Path("resnet101_embeddings.pt")
    PNP_IMGS_DATA_PATH = DATA_DIR / Path("pnp_img_data_with_paths_n_missing_imgs.csv")

    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    IMG_SIZE = (199,) * 2

    NUM_OF_PRODUCTS_RESULTS = 5
