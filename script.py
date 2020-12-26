#importing libraries
import os
import utils
from numpy import zeros
from model import baselineGRU
from beer_styles import encode_style
from flask import Flask, render_template, request

static_folder = os.path.join(os.pardir, 'static')
#creating instance of the class with static file location
app = Flask(__name__, static_url_path='/static')
    # prepare the model
model = baselineGRU()
model.zero_grad()
# each time we are only generating one character
model.hidden = model.init_hidden(1)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html',styles=encode_style, title="Hi", prediction="")

# shows the results of PyTorch model prediction
@app.route('/predict',methods=['POST'])
def predict():
    form_inputs = request.form
    if "beerstyle" not in form_inputs:
        return index()
    
    style = form_inputs['beerstyle']
    rate = form_inputs['rateInput']
    temp = float(form_inputs['temp'])

    dat = zeros([1,1,208])
    # OH encode beer style
    dat[0][0][encode_style[style]] = 1
    # OH encode rating
    dat[0][0][104 + int(rate) % 5] = 1
    # OH encode "Start of Sentence" char
    dat[0][0][110 + utils.char2pos('\x02')] = 1
    
    specs = (f"Style = {style}, "
              f"Rating = {rate}, "
              f"Temperature = {temp}: ")

    prediction = utils.generate_once(model, dat, temp) + '...'
    return render_template('index.html',styles=encode_style, title="Hi",
                           prediction=[specs, prediction])

if __name__ == '__main__':
    app.run(debug=False,port=os.getenv('PORT',5000))