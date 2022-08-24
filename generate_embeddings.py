import numpy as np
from PIL import Image
import torch
from pathlib import Path
from tqdm import tqdm
from torchvision import models
import torch.nn as nn
from torchvision import transforms as T
import uuid
import traceback
from common.common_utils import CommonUtils
from config import Config

import annoy
import argparse
import random


class GenerateEmbeddings:
    IMG_SIZE = (299, 299)
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]

    transforms = T.Compose(
        [T.ToTensor(), T.Resize(IMG_SIZE), T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)]
    )

    def __init__(
        self,
        img_list,
        model=None,
        embedding_size=2048,
        device="cpu",
    ):
        self.embedding_size = embedding_size
        self.img_list = img_list
        self.load_model(model, device)

    def load_model(self, model, device):
        _seed = Config.SEED
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(_seed)
        random.seed(_seed)
        np.random.seed(_seed)
        torch.cuda.manual_seed_all(_seed)

        if model:
            self.model = model
        else:
            self.model = models.resnet101(
                pretrained=True
                # weights='DEFAULT'
            )
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, self.embedding_size)
        self.model = self.model.to(device)
        self.model.eval()

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
            print("FileNotFoundError: ", img_path)
            return np.zeros(self.embedding_size)

        image_rgb = img.convert("RGB")
        transformed_img = self.transforms(image_rgb)

        if transformed_img.shape[0] == 4:
            transformed_img = CommonUtils.rgba2rgb(transformed_img)
        img_batch = transformed_img.unsqueeze(0)

        # Calculate embedding (unsqueeze to add batch dimension)
        img_embedding = self.model(img_batch)
        img_embedding_detached = (
            img_embedding.detach().numpy().reshape(self.embedding_size)
        )

        return img_embedding_detached

    def generate_embeddings(
        self,
    ):
        self.embeddings_list = []

        for im_path in tqdm(self.img_list, desc="Generating embeddings"):
            embeddings = dict()
            embeddings["im_path"] = im_path

            try:
                embeddings["embeddings"] = self.to_embedding(im_path)
            except:
                print("### ERROR  to_embedding ###")
                print(im_path)
                traceback.print_exc()
                print()
                print()

                embeddings["embeddings"] = np.zeros(self.embedding_size)
            self.embeddings_list.append(embeddings)

        return self.embeddings_list

    def generate_AnnoyIndex(self, distance_measure="manhattan", num_of_trees=100):
        self.t = annoy.AnnoyIndex(self.embedding_size, distance_measure)
        print()
        for n, i in enumerate(
            tqdm(self.embeddings_list, desc="Adding embeddings to annoy tree")
        ):
            self.t.add_item(n, i["embeddings"])
        self.t.build(num_of_trees)
        return self.t


def main(args):

    IMG_DIR = Path(args.img_dir_path)
    img_list = list(IMG_DIR.glob("*"))

    g = GenerateEmbeddings(img_list=img_list)

    # generate and save the embeddings
    embeddings = g.generate_embeddings()
    embeddings_save_path = (
        Config.EMBEDDINGS_PATH / f"{args.store_name}-{uuid.uuid4()}.pt"
    )
    torch.save(embeddings, embeddings_save_path)

    # index the embeddings and save the indexes
    t = g.generate_AnnoyIndex(
        num_of_trees=args.num_of_trees, distance_measure=args.distance_measure
    )
    t.save(Config.INDEXES_PATH.as_posix() + f"/{args.store_name}-{uuid.uuid4()}.ann")


if __name__ == "__main__":
    # python generate_embeddings.py -d ../img_dir_path  -n game_store
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--img_dir_path",
        help="The path to the image directory",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-n",
        "--store-name",
        default=str(uuid.uuid4()),
        help="The name of the store",
    )
    parser.add_argument(
        "--num-of-trees", default=100, type=int, help="The number of trees for annoy"
    )
    parser.add_argument(
        "--distance-measure",
        default="manhattan",
        help="The type of distance used to find closest embeddings",
    )
    args = parser.parse_args()

    main(args)
