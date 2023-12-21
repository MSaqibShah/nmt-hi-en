from flask import Flask, request, jsonify

import nemo
import nemo.collections.nlp as nemo_nlp
nmt_model = nemo_nlp.models.machine_translation.MTEncDecModel.from_pretrained(model_name="nmt_hi_en_transformer12x2")




app = Flask(__name__)


@app.route('/api_test', methods=['GET'])
def api_test():
    return jsonify({'message': 'NMT Hindi To English is working'})


@app.route('/translate', methods=['POST'])  
def translate():
    data = request.get_json(force=True)
    if not data:
        return jsonify({'message': 'No input data provided'}), 400
    
    text = data['text']

    if not text:
        return jsonify({'message': 'No text provided'}), 400    
    result = nmt_model.translate(source_text=text, source_lang="hi", target_lang="en")
    return jsonify(result)

@app.route('/translate_file', methods=['GET'])
def translate_file():
    with open('input.txt', 'rb') as f:
        text = f.read()
    result = nmt_model.translate(text, source_lang="hi", target_lang="en")
    return jsonify(result)
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5007)




