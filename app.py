from flask import Flask, request, jsonify
import testing_on_doc2vec
# Initialize the Flask app
app = Flask(__name__)

# GET and POST on the same route
@app.route('/classify-urls', methods=['GET'])
def execute_classify_urls():
    input_url = request.args.get('input_url')
    res = testing_on_doc2vec.classify_urls(input_url)
    return res


# Run the app
if __name__ == '_main_':
    app.run(debug=True)