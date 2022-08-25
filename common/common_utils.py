import base64
import io
from PIL import Image, UnidentifiedImageError
import numpy as np


class CommonUtils:
    @staticmethod
    def base64_2_pil(data):
        decoded = base64.b64decode(data)
        try:
            img = Image.open(io.BytesIO(decoded))
            return img
        except UnidentifiedImageError:
            print("UnidentifiedImageError")
            print(data)
            return None

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

    @staticmethod
    def crop_img(img_array, w_scale=0.70, h_scale=0.40):
        center_x, center_y = img_array.shape[1] / 2, img_array.shape[0] / 2
        width_scaled, height_scaled = (
            img_array.shape[1] * w_scale,
            img_array.shape[0] * h_scale,
        )
        left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
        top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
        img_cropped = img_array[int(top_y) : int(bottom_y), int(left_x) : int(right_x)]
        bounds = [left_x, top_y, right_x, bottom_y]
        return img_cropped, bounds
