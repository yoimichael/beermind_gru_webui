import os
import utils
import redis
from functools import partial
from model import baselineGRU
from beer_styles import encode_style
from flask import Flask, render_template, request

# creating instance of the class with static file location
app = Flask(__name__, static_url_path='/static')
r = redis.from_url(os.environ.get('REDIS_URL'))
# prepare the model
model = baselineGRU()
# partial "render_template" with default parameters for main UI
index_page = partial(render_template, 'index.html', styles=encode_style, title="Hi")


@app.route('/')
@app.route('/index')
def index():
    return index_page(prediction=[])


# shows the results of PyTorch model prediction
@app.route('/predict', methods=['POST'])
def predict():
    if "beerstyle" not in request.form:
        return index_page(prediction=[])

    style = request.form['beerstyle']
    rate = request.form['rateInput']
    temp = float(request.form['temp'])
    specs = (f"Style = {style}, "
             f"Rating = {rate}, "
             f"Temperature = {temp}: ")

    # use results from memcache
    memcache_key = f'{style}{rate}{temp}'
    prediction = r.get(memcache_key)
    if prediction and len(prediction) > 100:
        return index_page(prediction=[specs, prediction.decode("utf-8")])

    specs = (f"Style = {style}, "
             f"Rating = {rate}, "
             f"Temperature = {temp}: ")

    prediction = utils.generate_once(model, style, rate, temp) + '...'
    r.mset({memcache_key: prediction})
    return index_page(prediction=[specs, prediction])


if __name__ == '__main__':
    app.run(debug=False, port=os.getenv('PORT', 5000))
