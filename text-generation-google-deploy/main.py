from tensorflow import keras
import numpy as np
import json
from flask import Flask, request, jsonify


app =  Flask(__name__)

modelLSTM =  keras.models.load_model("modelLSTM10.h5")

maxlen = 10

chars = [' ','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

char_to_idx = {'u': 0,'n': 1,'c': 2,'o': 3,'v': 4,'e': 5,'r': 6,' ': 7,'t': 8,'h': 9,'s': 10,'a': 11,'m': 12,'i': 13,'g': 14,'p': 15,'l': 16,'y': 17,'k': 18,'w': 19,'d': 20,'b': 21,'f': 22,'j': 23,'q': 24,'x': 25,'z': 26}

idx_to_char = {0: 'u',1: 'n',2: 'c',3: 'o',4: 'v',5: 'e',6: 'r',7: ' ',8: 't',9: 'h',10: 's', 11: 'a', 12: 'm', 13: 'i',14: 'g',15: 'p',16: 'l',17: 'y',18: 'k',19: 'w',20: 'd',21: 'b',22: 'f',23: 'j',24: 'q',25: 'x',26: 'z'}


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)


def textGenerator(seed):

    generatedText = seed+' '

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        for i in range(100):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(seed):
                x_pred[0, t, char_to_idx[char]] = 1.

            preds = modelLSTM.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = idx_to_char[next_index]

            seed = seed[1:] + next_char

            generatedText += next_char

    return generatedText


@app.route("/seed/new",methods=["POST"])
def index():
    if request.data:
        seedDict = json.loads(request.data)
        textGen = textGenerator(seedDict['seed'])

        return jsonify({'text-generated':textGen}), 200
    else:
        return "400 BAD REGQUEST", 400


if __name__ == '__main__':
    app.debug = True
    app.run()