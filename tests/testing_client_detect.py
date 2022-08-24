from PIL import Image
from pathlib import Path
import base64
import json
import io
import sys
from requests import get, put, patch, post, delete
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
    if len(sys.argv) > 1:

        img_num = int(sys.argv[1])
    else:
        img_num = np.random.randint(0, 27)

    # img_list = Path("test_images").glob("*")
    img_list = Path("../../game_store_images/").glob("*")
    img = Image.open(list(img_list)[img_num])

    img.save("testing_input_image.jpg")

    data = json.dumps(
        {
            "img_data": pil2base64(img),
        }
    )

    res = post(
        TestConfig.base_url + "/detect",
        # "http://34.241.9.189:8000/detect",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    print("status_code: ", res.status_code)
    print("status_code: ", res.text)

    if res.ok:

        output = json.loads(res.text)

        print(output["label"])
        print()

        pprint(output["similar_products"])
