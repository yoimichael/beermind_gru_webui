import os
import utils
import redis
from rq import Queue
from worker import conn  # redis for bg tasks
from functools import partial
from model import baselineGRU
from beer_styles import encode_style
from flask import Flask, render_template, request

# creating instance of the class with static file location
app = Flask(__name__, static_url_path='/static')
# redis for caching model outputs
r = redis.from_url(os.environ.get('REDIS_URL'))
# redis for background tasks
q = Queue(connection=conn)
# prepare the model
model = baselineGRU()
# partial "render_template" with default parameters for main UI
index_page = partial(render_template, 'index.html', styles=encode_style, title="Hi")


@app.route('/')
@app.route('/index')
@app.route('/predict', methods=['GET'])
def index():
    return index_page()


# shows the results of PyTorch model prediction
@app.route('/predict', methods=['POST'])
def predict():
    if "beerstyle" not in request.form:
        return index_page()

    style = request.form['beerstyle']
    rate = request.form['rateInput']
    temp = float(request.form['temp'])
    specs = (f"Style = {style}, "
             f"Rating = {rate}, "
             f"Temperature = {temp}: ")

    # use results from memcache
    memcache_key = f'{style}{rate}{temp}'
    prediction = r.get(memcache_key)
    # only use results with more than 100 characters, else generate a new one
    if prediction and len(prediction) > 100:
        return index_page(prediction=[specs, prediction.decode("utf-8")])

    # make it an async job
    job = q.enqueue(utils.generate_once, model, style, rate, temp)
    # initialize output generation progress in the Job instance
    return index_page(job_id=job.get_id(), specs=specs)


@app.route('/result/<job_id>')
def get_job_result(job_id):
    job = q.fetch_job(job_id)
    style, rate, temp = job.get_call_string().split(', ')[-3:]
    style = style.strip("'")
    rate = rate.strip("'")
    temp = temp.strip(')')
    specs = (f"Style = {style}, "
             f"Rating = {rate}, "
             f"Temperature = {temp}: ")
    if job.result is None:
        return index_page(job_id=job.get_id(), specs=specs)

    predict_result = job.result + '...'
    memcache_key = f'{style}{rate}{temp}'

    # cache results in redis
    r.mset({memcache_key: predict_result})
    return index_page(prediction=[specs, predict_result])


if __name__ == '__main__':
    app.run(debug=False, port=os.getenv('PORT', 5000))
