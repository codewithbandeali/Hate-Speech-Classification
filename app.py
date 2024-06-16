
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/hatespeech', methods=['POST'])
def hatespeech():
    if request.method == 'POST':
        text = request.form['text']

        # Define paths

        model_path = 'C:\\Users\\PMLS\\Documents\\flaskMl\\hatespeech.h5'
        tokenizer_path = 'C:\\Users\\PMLS\\Documents\\flaskMl\\token.pkl'

        # Check if files exist
        if not os.path.exists(model_path):
            return render_template('index.html', result='Model file not found.')
        if not os.path.exists(tokenizer_path):
            return render_template('index.html', result='Tokenizer file not found.')

        try:
            # Load the model and tokenizer
            model = load_model(model_path)
            with open(tokenizer_path, 'rb') as file:
                tokenizer = pickle.load(file)

            # Tokenize the input text
            max_len = 25
            sequence = tokenizer.texts_to_sequences([text])
            padded = pad_sequences(sequence, maxlen=max_len, padding='post')

            # Predict the class
            pred = model.predict(padded)
            pred_class = np.argmax(pred,axis=-1)

            # Encode the result
            if pred_class == 0:
                result = 'Hate'
            elif pred_class == 1:
                result = 'Offensive'
            else:
                result = 'Neutral'

            return render_template('index.html', result=result)
        except Exception as e:
            return render_template('index.html', result=f'Error: {str(e)}')

    return render_template('index.html', result='Method not allowed.')

if __name__ == '__main__':
    app.run(debug=True)



