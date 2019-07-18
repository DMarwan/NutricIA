import numpy as np
import pandas as pd
import os, sys


# Modelling
import keras
from keras.layers import Dropout, Flatten,Activation, Dense

from keras.optimizers import SGD
from keras import backend as K
from keras import applications
from keras.preprocessing.image import array_to_img, img_to_array, load_img, image
from nutricia import make_labels, cal_table

import pickle

# API

from flask import Flask, render_template, url_for, redirect, request


app = Flask(__name__)


# Predict the clas name and the likelihood of each class
def classify(imgarray, loaded_model):
	class_names = make_labels()
	preds = np.round(loaded_model.predict(imgarray),2)

	predict_index = np.argmax(preds)
	predict_proba = preds[0][predict_index]
	predict = class_names[predict_index]

	final = pd.DataFrame({'Food Type' : np.array(class_names),'Likelihood' :preds[0]})

	K.clear_session()
	return final.sort_values(by = 'Likelihood',ascending=False).head(), predict



@app.route('/')
def landingpage():
	return render_template('index.html')

@app.route('/upload')
def upload():
	return render_template('upload.html')

@app.route('/analyse', methods=['GET','POST'])
def predict():
	if request.method == 'POST':
		K.clear_session()

		filename = 'model/modelFood.pkl' #VGG-16
		loaded_model = pickle.load(open(filename, 'rb'))

		print("\n\nLoaded Model from disk\n\n")

		foodpred =  request.files['meal']
		imgtest = load_img(foodpred,target_size=(224,224))
		imgarray = img_to_array(imgtest)
		imgarray = imgarray/255
		imgarray = imgarray.reshape(1,imgarray.shape[0],imgarray.shape[1],imgarray.shape[2])



		final, pred_class = classify(imgarray, loaded_model)
		pred_class = pred_class.lower()
		return render_template("analyse.html", tables=[final.to_html(classes='calories', index = False, justify='left')], calories = [cal_table(pred_class).to_html(classes='calories', justify='left')], titles=final.columns.values, name = pred_class, foodpred = imgtest)


@app.route('/dashboard')
def dashboard():
	return render_template('dashboard.html')

@app.route('/charts')
def charts():
	return render_template('charts.html')

@app.route('/tables')
def tables():
	return render_template('tables.html')

if __name__ == '__main__':
    app.run(host = '0.0.0.0',port=1200, debug =  True)
