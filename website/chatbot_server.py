from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
from tensorflow import keras
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import random
import os

app = Flask(__name__)
CORS(app)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Path to UniBot-Demo project
UNIBOT_PATH = "../UniBot-Demo"  # Change this to your UniBot-Demo folder path

# Load the trained model and required files from UniBot-Demo
try:
    # Load the model
    model = keras.models.load_model(os.path.join(UNIBOT_PATH, 'model.h5'))
    print("Model loaded successfully")

    # Load the intents
    with open(os.path.join(UNIBOT_PATH, 'intents.json')) as file:
        intents = json.load(file)
    print("Intents loaded successfully")

    # Load words and classes
    with open(os.path.join(UNIBOT_PATH, 'words.pkl'), 'rb') as file:
        words = pickle.load(file)
    print("Words loaded successfully")

    with open(os.path.join(UNIBOT_PATH, 'classes.pkl'), 'rb') as file:
        classes = pickle.load(file)
    print("Classes loaded successfully")

except Exception as e:
    print(f"Error loading files: {str(e)}")
    print("Make sure the UniBot-Demo folder path is correct and contains required files")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list):
    if not intents_list:
        return "I'm not sure I understand. Could you please rephrase that?"
    
    tag = intents_list[0]['intent']
    list_of_intents = intents['intents']
    
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            return result
    
    return "I'm not sure how to respond to that."

@app.route('/get', methods=['GET'])
def get_bot_response():
    try:
        user_message = request.args.get('msg')
        if not user_message:
            return "Please send a message."

        ints = predict_class(user_message)
        response = get_response(ints)
        return response

    except Exception as e:
        print(f"Error: {str(e)}")
        return "I encountered an error. Please try again."

@app.route('/')
def home():
    return "UniBot Server is running!"

if __name__ == "__main__":
    # Download required NLTK data
    try:
        nltk.download('punkt')
        nltk.download('wordnet')
        print("NLTK data downloaded successfully")
    except Exception as e:
        print(f"Error downloading NLTK data: {str(e)}")

    app.run(debug=True, port=5000) 