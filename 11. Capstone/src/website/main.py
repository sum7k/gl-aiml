from flask import Flask
from flask import request
from flask import render_template, send_from_directory, url_for

import sys, os, cv2
from pathlib import Path

from .model import Model
from .helper import draw_boxes


model = Model()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/home/ubuntu/website/static/upload/'


if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    model.initialize()    

@app.route('/', methods=['GET', 'POST'])
def index():
        print(request.method, flush=True)
        print(request, flush=True)
        if request.method == 'POST':

            uploaded_file = request.files['dicom']
            uploaded_file_path = app.config['UPLOAD_FOLDER'] + uploaded_file.filename
            uploaded_file.save(uploaded_file_path)

            image, result = model.predict(uploaded_file_path)
            draw_boxes(image, result)

            output_file_name = Path(uploaded_file_path).stem + ".jpg"
            output_file_path = app.config['UPLOAD_FOLDER'] + output_file_name
            cv2.imwrite(output_file_path, image)


            print("Result:", result)
            sys.stdout.flush()
            return render_template('index.html', result = result, image_file = url_for('uploaded_file', filename=output_file_name))
        else:
            return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)