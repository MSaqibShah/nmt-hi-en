from flask import Flask, request, jsonify

import nemo
import nemo.collections.nlp as nemo_nlp
nmt_model = nemo_nlp.models.machine_translation.MTEncDecModel.from_pretrained(model_name="nmt_hi_en_transformer12x2")




app = Flask(__name__)


@app.route('/api_status', methods=['GET'])
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

    if type(text) != list:
        return jsonify({'message': 'Text should be a list'}), 400
    
    if len(text) > 1000:
        return jsonify({'message': 'Text too long'}), 400

    response = []
    for line in text:
        result = nmt_model.translate(line, source_lang="hi", target_lang="en")
        response.append(result)

    with open('output.txt', 'w',encoding='utf-8') as f:
        for item in response:
            item = " ".join(item)
            f.write("%s\n" % item)

    return jsonify(response)

@app.route('/translate_file', methods=['GET'])
def translate_file():
    with open('input.txt', 'r',encoding='utf-8') as f:
        text = f.readlines()

    response = []
    for line in text:
        result = nmt_model.translate(line, source_lang="hi", target_lang="en")
        response.append(result)
        
    with open('output.txt', 'w',encoding='utf-8') as f:
        for item in response:
            item = " ".join(item)
            f.write("%s\n" % item)


    return jsonify("Translation Saved in output.txt")
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5007)




