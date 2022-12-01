# Imporing Libraries
from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import random
import sys
import io


########################################
############# LSTM   ###################
########################################


def create_dictionary(text):

    char_to_idx = dict()
    idx_to_char = dict()

    idx = 0
    for char in text:
        if char not in char_to_idx.keys():
            # Build dictionaries
            char_to_idx[char] = idx
            idx_to_char[idx] = char
            idx += 1

    return char_to_idx, idx_to_char

def read_dataset(filename):

    letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m',
            'n','o','p','q','r','s','t','u','v','w','x','y','z',' ']
    
    # Open raw file
    with open(filename, 'r') as f:
        raw_text = f.readlines()

    # Removing the default column
    raw_text.pop(0)

    # Transform each line into lower
    raw_text = [line.lower() for line in raw_text]
    
    # Create a string which contains the entire text
    text_string = ''
    for line in raw_text:
        text_string += line.strip()

    print('corpus length:',len(text_string))

    # Create an array by char
    text = list()
    for char in text_string:
        text.append(char)

    # Remove all symbosl and just keep letters
    text = [char for char in text if char in letters]

    # Getting all characters in the text
    chars = sorted(list(set(text_string)))

    print('total chars:',len(chars))

    # Getting the idx of all char in the text
    char_to_idx, idx_to_char = create_dictionary(text)

    return text_string,char_to_idx,idx_to_char,chars

def sequenceGen(text,maxlen):
    step = 3
    sentences = []
    next_chars = []

    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])

    print('nb sequences:', len(sentences))

    return sentences,next_chars

def vectorization(chars,maxlen,sentences,char_to_idx,next_chars):

    print('Vectorization...')
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_to_idx[char]] = 1
        y[i, char_to_idx[next_chars[i]]] = 1

    return x,y

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    
    return np.argmax(probas)

def modelLSTM(maxlen,chars):
    print('Build model...')

    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars), activation='softmax'))
    optimizer = RMSprop(learning_rate=0.01)
    
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])

    return model

#################################################
##############  KERAS CALLBACK ##################
#################################################


class EarlyTextGen(keras.callbacks.Callback):

    def __init__(self,model,text,maxlen,chars,char_to_idx,idx_to_char):
      super(EarlyTextGen, self).__init__()
      self.model = model
      self.text = text
      self.maxlen = maxlen
      self.chars = chars
      self.char_to_idx = char_to_idx
      self.idx_to_char = idx_to_char
  

    def on_epoch_end(self,epoch,logs=None):
        # Function invoked at end of each epoch. Prints generated text.
        print()
        print('----- Generating text after Epoch: %d' % epoch)

        start_index = random.randint(0, len(self.text) - self.maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('----- diversity:', diversity)

            generated = ''
            sentence = self.text[start_index: start_index + self.maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(400):
                x_pred = np.zeros((1, self.maxlen, len(self.chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, self.char_to_idx[char]] = 1.

                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = self.idx_to_char[next_index]

                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()



text_string,char_to_idx,idx_to_char,chars = read_dataset('data/nlp-v2.txt')

maxlen = 10
sentences,next_chars= sequenceGen(text_string,maxlen=maxlen)

x,y = vectorization(chars,maxlen,sentences,char_to_idx,next_chars)

MODEL = modelLSTM(maxlen,chars)

model_weights_path = f"results/modelLSTM-Epoch"+str(20)+".h5"

r = MODEL.fit(x, y,
          batch_size=200,
          epochs=20,
          callbacks=[EarlyTextGen(MODEL,text_string,maxlen,chars,char_to_idx,idx_to_char)])


MODEL.save("my_model.h5")