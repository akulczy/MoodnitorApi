# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 20:46:07 2021

@author: agata
"""


import os
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from tensorflow import keras
import numpy as np

app = Flask(__name__)
api = Api(app)
import tensorflow as tf

from tensorflow.keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import nltk.data

@app.route('/', methods=['GET'])
def home():
   return "<h2>API created for the Moodnitor project.</h2>"

class Predict(Resource):
    @staticmethod
    def post():
        
        nltk_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        # loading tokenizer
        import pickle
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
    
        data = request.get_json()
        emotion_data = data['emotion']
        
        sentences = nltk_tokenizer.tokenize(emotion_data)
        
        # loading pre-trained model
        m = tf.keras.models.load_model('emotionmodel.hdf5')
    
        class_names = ['joy', 'fear', 'anger', 'sadness', 'neutral']
        
        predictions = []
        
        # Splitting entry into sentences
        for emotion in sentences:        
            
            text_txt_tokenized = tokenizer.texts_to_sequences([emotion])
        
            prediction_sentence = []
            for item in text_txt_tokenized:
                for i in item:
                    prediction_sentence.append(i)
                    
            prediction_sentence = [prediction_sentence]
            
            padded_item = pad_sequences(prediction_sentence, padding='post', maxlen=125)
            
            prediction = m.predict(padded_item)
                    
            pred_class = class_names[np.argmax(prediction)]
            
            pred_class_value = max(prediction[0])
            
            # object to be returned in the reponse
            obj = {
                "sentence": emotion,
                "prediction": prediction.tolist(),
                "predclass": pred_class,
                "maxVal": str(pred_class_value)
            }
            
            predictions.append(obj)   
            
        # jsonify method user to return response
        return jsonify(
            predictions = predictions
        )
    
api.add_resource(Predict, '/predict')
if __name__ == '__main__':
    app.run(host="localhost", port=5000)