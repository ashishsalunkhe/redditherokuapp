from flask import Flask, render_template, request, jsonify

from predict.file import predict_file
from predict.url import predict_url

app = Flask(__name__, template_folder='templates')


# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('main.html', result={"result": False})

    if request.method == 'POST':
        prediction = predict_url(request.form['url'])
        return render_template('main.html', result=prediction)


@app.route('/automated_testing', methods=['POST'])
def getfile():
    upload_file = request.files["upload_file"]
    data = upload_file.read()
    s = data
    line = []
    line = s.splitlines()
    dictionary = {}
    for i in line:
        string1 = str(i)
        string2 = string1[2:-1]
        string3 = predict_file(string2)
        dictionary[string2] = str(string3)

    return jsonify(dictionary)


if __name__ == '__main__':
    app.run()
