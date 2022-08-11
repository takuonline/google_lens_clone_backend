import base64
import io
from PIL import Image


class CommonUtils:
    @staticmethod
    def base64_2_pil(  data):
        decoded = base64.b64decode(data)
        img = Image.open(io.BytesIO(decoded))
        return img

    @staticmethod
    def pil2base64(im):
        buffered = io.BytesIO()
        im.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
