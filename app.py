from flask import Flask
from flask_restful import Api
from resources.detect import Detect
from resources.home import Home
from resources.search import ImgSearch

## preload resnet with trained weights


app = Flask(__name__)
api = Api(app)


api.add_resource(Home, "/")
api.add_resource(Detect, "/detect", endpoint="detect")
api.add_resource(ImgSearch, "/search", endpoint="search")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
