import pandas as pd
import annoy

import torch.nn as nn
from torchvision import models
from torchvision import transforms as T

from config import Config
from common.common_utils import CommonUtils

import numpy as np
import torch
import random

transforms = T.Compose(
    [
        T.ToTensor(),
        T.Resize(Config.IMG_SIZE),
        T.Normalize(Config.NORMALIZE_MEAN, Config.NORMALIZE_STD),
    ]
)


class ImgSearchModel:
    def __init__(
        self,
        stored_img_indexes_path,
        img_metadata,
        device,
        embedding_size=2048,
        model=None,
    ):
        self.embedding_size = embedding_size

        _seed = Config.SEED
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(_seed)
        random.seed(_seed)
        np.random.seed(_seed)
        torch.cuda.manual_seed_all(_seed)

        # instantiate model
        if model:
            self.embedding_model = model
        else:
            self.embedding_model = models.resnet101(
                pretrained=True
                # weights='DEFAULT'
            )
        in_features = self.embedding_model.fc.in_features
        self.embedding_model.fc = nn.Linear(in_features, self.embedding_size)
        self.embedding_model = self.embedding_model.to(device)
        self.embedding_model.eval()

        # instatiate annoy tree
        self.tree = annoy.AnnoyIndex(embedding_size, "manhattan")
        self.tree.load(stored_img_indexes_path.as_posix())

        # load metadata
        self.df = pd.read_csv(img_metadata)
        if self.df.get("price", None).any():
            self.df["price"] = self.df["price"].fillna(0)

    def _transform(self, pil_img, rotate=False):

        if rotate:
            print("Rotating image ")
            img = img.rotate(270, expand=1)

        image_rgb = pil_img.convert("RGB")
        transformed_img = transforms(image_rgb)

        img_batch = transformed_img.unsqueeze(0)

        img_embedding = self.embedding_model(img_batch)
        img_embedding_detached = (
            img_embedding.detach().numpy().reshape(self.embedding_size)
        )

        return img_embedding_detached

    def _search_match(self, query_embedding, n: int = 5):
        res_indexes = self.tree.get_nns_by_vector(query_embedding, n=n)

        # ensures that the results are returned in the right order
        return pd.concat(
            [self.df[self.df["embedding_index"] == i] for i in res_indexes], axis=0
        )

    def search(self, detection_res: dict, num_of_results: int = 5):
        search_img = detection_res.get("search_img")
        if isinstance(search_img, str):
            pil_img = CommonUtils.base64_2_pil(search_img)
        else:
            pil_img = search_img

        del detection_res["search_img"]

        if pil_img:
            pil_img.save("tests/yolo_output.jpg")

        query_embedding = self._transform(pil_img)
        res = self._search_match(query_embedding, n=int(num_of_results))

        detection_res["similar_products"] = res.to_dict(orient="records")

        return detection_res
