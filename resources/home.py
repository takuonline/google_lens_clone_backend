from flask_restful import Resource


class Home(Resource):
    def get(self):
        return "Hello world", 200

    def post(self):
        pass
