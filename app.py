import os

from flask import Flask,request,render_template
from PIL import Image

from Loading_model import trial


app = Flask(__name__) #name of the application is app


@app.route('/', methods=['POST','GET'])
def upload():
    if request.method=='POST':
        f = request.files['file']        
        f.save(f.filename)
        predicted = trial(f.filename)
        os.remove(f.filename)        
        return render_template('file_upload.html',name=f.filename, predicted_value=predicted)        
    return render_template('file_upload.html')


if __name__== "__main__":
    app.run(host="localhost", port=1234, debug=True)