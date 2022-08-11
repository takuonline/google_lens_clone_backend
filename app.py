from flask import Flask
from flask_restful import Api
from resources.detect import Detect
from resources.home import Home


app = Flask(__name__)
api = Api(app)


api.add_resource(Home, "/")
api.add_resource(Detect, "/detect", endpoint="detect")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
