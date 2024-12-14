from flask import Flask, request, jsonify
import openai

app = Flask(__name__)

openai.api_key = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"

@app.route('/watson', methods=['POST'])
def ask_llm():
    user_input = request.json.get('question')
    response = openai.Completion.create(
        engine="your-model-name", 
        prompt=user_input,
        max_tokens=100
    )
    return jsonify({"response": response.choices[0].text.strip()})

if __name__ == '__main__':
    app.run(debug=True)
