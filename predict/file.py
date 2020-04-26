from model.predict import pred
from werkzeug.utils import secure_filename
import flask
from flask import Flask, render_template, request, make_response, jsonify

import os
import json

app = flask.Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = 'file_upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def predict_file(file):
    obj = {}
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        f = open(os.path.join(app.config['UPLOAD_FOLDER'], filename), "r")
        for line in f.read().splitlines():
            obj[line] = pred(line)['result']
            print(obj[line])
        # with open(os.path.join(app.config['UPLOAD_FOLDER'], 'resp.json'), 'w') as json_file:
        data = make_response(jsonify(obj))
        return {"data": data, "success": True}
