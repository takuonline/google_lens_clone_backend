import base64
import io
from PIL import Image,UnidentifiedImageError
import numpy as np

class CommonUtils:
    @staticmethod
    def base64_2_pil(data):
        decoded = base64.b64decode(data)
        try:
            img = Image.open(io.BytesIO(decoded))
            return img
        except UnidentifiedImageError:
            print(data)


    @staticmethod
    def pil2base64(im):
        buffered = io.BytesIO()
        im.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    @staticmethod
    def rgba2rgb(png):
        png = Image.fromarray(png)
        png.load()  # required for png.split()
        background = Image.new("RGB", png.size, (255, 255, 255))
        background.paste(png, mask=png.split()[3])  # 3 is the alpha channel

        return np.array(background)
