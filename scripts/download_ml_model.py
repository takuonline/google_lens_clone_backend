from torchvision import models


if __name__ == "__main__":
    # instantiate model
    embedding_model = models.resnet101(
        pretrained=True
        # weights='DEFAULT'
    )
