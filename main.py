# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START gae_python38_app]
# [START gae_python3_app]
from flask import Flask, render_template, request
# from werkzeug.utils import secure_filename
import os
from google.cloud import storage




# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`. 
app = Flask(__name__)

# UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'tmp/test')
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# CLOUD_STORAGE_BUCKET = os.environ['CLOUD_STORAGE_BUCKET']
CLOUD_STORAGE_BUCKET = "where-am-i-mit.appspot.com"



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods = ['GET', 'POST'])
def classify(): 

    """Process the uploaded file and upload it to Google Cloud Storage."""
    uploaded_file = request.files.get('file')

    if not uploaded_file:
        return 'No file uploaded.', 400

    # Create a Cloud Storage client.
    gcs = storage.Client()

    # Get the bucket that the file will be uploaded to.
    bucket = gcs.get_bucket(CLOUD_STORAGE_BUCKET)

    # Create a new blob and upload the file's content.
    blob = bucket.blob(uploaded_file.filename)

    blob.upload_from_string(
        uploaded_file.read(),
        content_type=uploaded_file.content_type
    )
  

    model_prediction = blob.public_url

      # once i have my prediction i can delete the file
    #   os.remove(filepath)

    return render_template('test.html', prediction=model_prediction)


@app.route('/go_back')
def go_back():
    return render_template('index.html')

if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_python3_app]
# [END gae_python38_app]
