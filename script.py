#importing libraries
import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request, send_from_directory
from beer_styles import encode_style
import utils
from model import baselineGRU
from numpy import zeros

static_folder = os.path.join(os.pardir, 'static')
#creating instance of the class with static file location
app = Flask(__name__, static_url_path='/static')
model = baselineGRU()

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html',styles=encode_style, title="Hi", prediction="")

# shows the results of PyTorch model prediction
@app.route('/predict',methods=['POST'])
def predict():
    #your application logic here
    data = request.form
    if "beerstyle" not in data:
        return index()
    
    
    dat = zeros([1,1,208])
    dat[0][0][int(data['beerstyle'])] = 1
    dat[0][0][104 + int(data['rateInput']) % 5] = 1
    dat[0][0][110:] = utils.char2oh('\x02')
    prediction = utils.generate(model, dat, float(data['temp']))[0] + "..."
    return render_template('index.html',styles=encode_style, title="Hi", prediction=prediction)
    

@app.route('/paper')
def read_paper():
    return send_from_directory("./", "paper.pdf", as_attachment=True)
    
if __name__ == '__main__':
    app.run(debug=False,port=os.getenv('PORT',5000))