from flask import Flask, request, jsonify

import nemo
import nemo.collections.nlp as nemo_nlp
nmt_model = nemo_nlp.models.machine_translation.MTEncDecModel.from_pretrained(model_name="nmt_hi_en_transformer12x2")


app = Flask(__name__)


@app.route('/api_status', methods=['GET'])
def api_test():
    return jsonify({'message': 'NMT Hindi To English is working'})


@app.route('/translate', methods=['GET'])
def translate_file():
    with open('input.txt', 'r',encoding='utf-8') as f:
        text = f.readlines()



    response = []
    for line in text:
        line = line.replace('\n','')
        line = line.replace('\r','')
        temp = f"{line}"
        result = nmt_model.translate(temp, source_lang="hi", target_lang="en")
        response.append(result)
        
    with open('output.txt', 'w',encoding='utf-8') as f:
        for item in response:
            item = " ".join(item)
            f.write("%s\n" % item)


    return jsonify({"message":"Translation Saved in output.txt"}),200

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5007)




