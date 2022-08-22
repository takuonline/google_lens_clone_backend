import pandas as pd
import numpy as np

import torch.nn as nn
from torchvision import models
from torchvision import transforms as T
import torch

import torchvision.transforms.functional as TVF

from scipy import spatial

from config import Config
from common.common_utils import CommonUtils


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
        stored_img_embeddings_path,
        img_metadata,
        device,
        embedding_size=2048,
        model=None,
    ):
        self.embedding_size = embedding_size

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
        _ = self.embedding_model.eval()

        # instatiate kdtree

        stored_img_embeddings = torch.load(stored_img_embeddings_path)
        self.tree = spatial.KDTree(stored_img_embeddings)

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

        if transformed_img.shape[-1] == 4:
            transformed_img = transformed_img[:, :, :3]

        img_batch = transformed_img.unsqueeze(0)

        img_embedding = self.embedding_model(img_batch)
        img_embedding_detached = (
            img_embedding
            .detach()
            .numpy()
            .reshape(self.embedding_size)
        )

        return img_embedding_detached

    def _search_match(self, query_embedding, k=5):

        v = self.tree.query(query_embedding, k=k)

        res_indexes = list(v[1][:])
        # print(res_indexes)
        return self.df.iloc[res_indexes]

    def run_search(self, detection_res: dict, num_of_results: int = 5):
        pil_img = detection_res["output_img"]
        # print("run_search"*30)
        pil_img.save("run_searchoutput.jpg")
        query_embedding = self._transform(pil_img)
        res = self._search_match(query_embedding, k=int(num_of_results))

        detection_res["similar_products"] = res.to_dict(orient="records")
        # detection_res["output_img"] = CommonUtils.pil2base64(pil_img)
        # print(detection_res)
        del detection_res["output_img"]

        return detection_res
