import pandas as pd
import numpy as np

import torch.nn as nn
from torchvision import models
from torchvision import transforms as T
import torch

import torchvision.transforms.functional as TVF

from scipy import spatial

from config.config import Config
from common.common_utils import CommonUtils


transforms = T.Compose(
    [
        T.ToTensor(),
        T.Resize(Config.IMG_SIZE),
        T.Normalize(Config.NORMALIZE_MEAN, Config.NORMALIZE_STD),
    ]
)


class Resnet101Search:
    num_classes = 2048

    def __init__(self, stored_img_embeddings_path,device,img_metadata):

        # instantiate model
        self.embedding_model = models.resnet101(weights='DEFAULT')
        in_features = self.embedding_model.fc.in_features
        self.embedding_model.fc = nn.Linear(in_features, self.num_classes)
        self.embedding_model = self.embedding_model.to(device)
        _ = self.embedding_model.eval()

        # instatiate kdtree
        print(stored_img_embeddings_path)
        stored_img_embeddings = torch.load(stored_img_embeddings_path)
        self.tree = spatial.KDTree(stored_img_embeddings)

        # load pnp metadata
        self.pnp_df = pd.read_csv(img_metadata)
        self.pnp_df["price"] = self.pnp_df["price"].fillna(0)

    def transform(self, pil_img, rotate=False):

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
            img_embedding.detach().numpy().reshape(self.num_classes)
        )

        return img_embedding_detached

    def search_match(self, query_embedding, k=5):

        v = self.tree.query(query_embedding, k=k)
        res_indexes = list(v[1][:])
        return self.pnp_df.iloc[res_indexes]

    def run_search(self, detection_res:dict,num_of_results:int=5):
        pil_img = detection_res["output_img"]
        pil_img.save("output.jpg")
        img_embedding_detached = self.transform(pil_img)
        res = self.search_match(img_embedding_detached, k=int(num_of_results))

        detection_res["similar_products"] = res.to_dict(orient="records")
        detection_res["output_img"] = CommonUtils.pil2base64(pil_img)

        return detection_res
