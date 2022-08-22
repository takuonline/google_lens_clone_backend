import pandas as pd

import numpy as np
from PIL import Image
import torch
from pathlib import Path
from tqdm import tqdm
from torchvision import models
import torch.nn as nn
from torchvision import transforms as T
import uuid

from common.common_utils import CommonUtils
from config import Config

class GenerateEmbeddings:
    IMG_SIZE = (299, 299)
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]

    transforms = T.Compose(
        [T.ToTensor(), T.Resize(IMG_SIZE), T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)]
    )

    def __init__(self, model=None, num_classes=2048, device="cpu"):
        self.num_classes = num_classes
        self.load_model(model, device)

    def load_model(self, model, device):
        if model:
            self.model = model
        else:
            self.model = models.resnet101(
                pretrained=True
                # weights='DEFAULT'
            )
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, self.num_classes)
        self.model = self.model.to(device)
        _ = self.model.eval()

    def rgba2rgb(self, png):
        png = Image.fromarray(png)
        png.load()  # required for png.split()
        background = Image.new("RGB", png.size, (255, 255, 255))
        background.paste(png, mask=png.split()[3])  # 3 is the alpha channel
        return np.array(background)

    def to_embedding(self, img_path, rotate=False):
        try:
            img = Image.open(img_path)
            if rotate:
                print("Rotating image ")
                img = img.rotate(270, expand=1)

        except FileNotFoundError:
            return np.zeros(self.num_classes)

        image_rgb = img.convert("RGB")

        transformed_img = self.transforms(image_rgb)

        if transformed_img.shape[2] == 4:
            transformed_img = CommonUtils.rgba2rgb(transformed_img)

        try:
            img_batch = transformed_img.unsqueeze(0)
            # Calculate embedding (unsqueeze to add batch dimension)
            img_embedding = self.model(img_batch)
            img_embedding_detached = (
                img_embedding.detach().numpy().reshape(self.num_classes)
            )
        except ValueError:
            print(img_path)
            return np.zeros(self.num_classes)

        return img_embedding_detached


if __name__ == "__main__":
    # Exception("Please set the img_dir path and the emb save path")

    # IMG_DIR = Path("../4. woolies_images/all_imgs")
    IMG_DIR = Path("../game_store_images")

    img_list = list(IMG_DIR.glob("*"))
    g = GenerateEmbeddings()

    embeddings = [g.to_embedding(im_path) for im_path in tqdm(img_list)]
    embeddings_save_path = Config.EMBEDDINGS_PATH / f"{uuid.uuid4()}-game.pt"

    torch.save(embeddings, embeddings_save_path)
