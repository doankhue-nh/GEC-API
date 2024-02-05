from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import components.gector.predict as gector

app = Flask(__name__)
api = Api(app)

_ROBERTA_GPU = 0
model_gector_roberta = gector.load_for_demo(use_roberta=True, gpu_id=_ROBERTA_GPU)

class MODEL(Resource):
    # def get(self):
    #     return jsonify({"message": "This endpoint supports GET requests."})
    def post(self):
        json_data = request.get_json(force=True)
        model = json_data['model']
        text_input_list = json_data['text_input_list']
        print(f'======INPUT TO {model}=====', flush=True)
        print(text_input_list, sep='\n', flush=True)
        if model == 'GECToR-Roberta':
            text_output_list = gector.predict_for_demo(text_input_list, model_gector_roberta)
        else:
            raise NotImplementedError(f'Model {model} is not recognized.')        
        print(f'======OUTPUT FROM {model}=====', flush=True)
        print(text_output_list, sep='\n', flush=True)
        return jsonify({'model': model, 'text_output_list': text_output_list})

api.add_resource(MODEL, "/components/model", methods=['GET', 'POST'])


if __name__ == '__main__':
    #app.run(host='127.0.0.1', port=5000, debug=True)
    app.debug = True
    app.run(host='0.0.0.0', port=3000, use_reloader=False)