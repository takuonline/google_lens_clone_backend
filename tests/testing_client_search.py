from PIL import Image
import base64
import json
import io
from requests import get, put, patch, post
from pprint import pprint
import numpy as np
from test_config import TestConfig


def base64_2_pil(data):
    decoded = base64.b64decode(data)
    img = Image.open(io.BytesIO(decoded))
    return img


def pil2base64(im):
    buffered = io.BytesIO()
    im.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


if __name__ == "__main__":

    img_list = list(TestConfig.img_path.glob("*"))
    img_num = np.random.randint(0, len(img_list))
    print("rand int", img_num)

    img = Image.open(img_list[img_num])

    img.save("testing_input_image.jpg")

    data = json.dumps({"search_img": pil2base64(img)})

    res = post(
        TestConfig.base_url + "/search",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    print("status_code: ", res.status_code)

    if res.ok:
        output = json.loads(res.text)
        pprint(output["similar_products"])
