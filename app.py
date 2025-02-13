from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "./keras model saved/TSR/"
model = load_model(MODEL_PATH)

# Define the classes for traffic signs
classes = {
    0:'Speed limit (5km/h)', 1:'Speed limit (15km/h)', 2:'Speed limit (30km/h)', 3:'Speed limit (40km/h)',
    4:'Speed limit (50km/h)', 5:'Speed limit (60km/h)', 6:'Speed limit (70km/h)', 7:'Speed limit (80km/h)',
    8:'Dont Go straight or left', 9:'Dont Go straight or Right', 10:'Dont Go straight', 11:'Dont Go Left',
    12:'Dont Go Left or Right', 13:'Dont Go Right', 14:'Dont overtake from Left', 15:'No Uturn',
    16:'No Car', 17:'No horn', 18:'Speed limit (40km/h)', 19:'Speed limit (50km/h)',
    20:'Go straight or right', 21:'Go straight', 22:'Go Left', 23:'Go Left or right',
    24:'Go Right', 25:'keep Left', 26:'keep Right', 27:'Roundabout mandatory',
    28:'watch out for cars', 29:'Horn', 30:'Bicycles crossing', 31:'Uturn',
    32:'Road Divider', 33:'Traffic signals', 34:'Danger Ahead', 35:'Zebra Crossing',
    36:'Bicycles crossing', 37:'Children crossing', 38:'Dangerous curve to the left',
    39:'Dangerous curve to the right', 40:'Unknown1', 41:'Unknown2', 42:'Unknown3',
    43:'Go right or straight', 44:'Go left or straight', 45:'Unknown4', 46:'ZigZag Curve',
    47:'Train Crossing', 48:'Under Construction', 49:'Unknown5', 50:'Fences',
    51:'Heavy Vehicle Accidents', 52:'Unknown6', 53:'Give Way', 54:'No stopping',
    55:'No entry', 56:'Unknown7', 57:'Unknown8'
}

# Function to preprocess and predict traffic sign
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = Image.open(file)
            img = img.resize((30, 30))
            img = np.array(img)
            img = np.expand_dims(img, axis=0)  # Reshape to match model input
            
            predict_x = model.predict(img)
            Y_pred = np.argmax(predict_x, axis=1)
            prediction = classes[Y_pred[0]]
            
            return render_template('index.html', prediction=prediction)
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
