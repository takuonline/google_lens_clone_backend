from PIL import Image
from pathlib import Path
import base64
import json
import io
import sys
from requests import get, put, patch, post



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
    img_num = int(sys.argv[1] )
    img_list = Path("test_images").glob("*")
    img = Image.open(list(img_list)[img_num])

    img.save("output.jpg")
 
    data = json.dumps({"img_data": pil2base64(img)})

    res = post(
        "http://localhost:5000/detect",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    print("status_code: ", res.status_code)
    if res.ok:
        output = json.loads(res.text)

        print(output["label"])
        print()

        output_img = base64_2_pil(output["output_img"])
        output_img.save("client_side_output.jpg")


