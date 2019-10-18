from __future__ import print_function
from keras.callbacks import LambdaCallback, CSVLogger
from keras.models import Sequential
from keras.layers import Dense, Input, concatenate
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.utils.data_utils import get_file
from keras.models import Model
import numpy as np
import random
import sys
import io

path = get_file('C:\\MLProjects\\venv7\\pasta.txt',origin='')
with io.open(path) as f:
    text = f.read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
num_chars = len(chars)

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
        y[i, t, char_indices[next_chars[i]]] = 1

vec = Input(shape=(None, num_chars))
l_1 = LSTM(128, dropout=0.2, return_sequences=True)(vec)
input2 = concatenate([vec, l_1], axis=-1)
l_2 = LSTM(128, dropout=0.2, return_sequences=True)(input2)
input3 = concatenate([vec,l_2], axis=-1)
l_3 = LSTM(128, dropout=0.2, return_sequences=True)(input3)
inputd = concatenate([l_1,l_2,l_3], axis=-1)
dense = Dense(num_chars, activation='softmax')(inputd)
model = Model(inputs=[vec], outputs=[dense])

model.compile(loss='categorical_crossentropy', optimizer=Adam(clipnorm=1.), metrics=['accuracy'])

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    if epoch % 10 == 0:
    # Function invoked at end of each epoch. Prints generated text.
        print()
        print('----- Generating text after Epoch: %d' % epoch)

        start_index = random.randint(0, len(text) - maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('----- diversity:', diversity)

            generated = ''
            sentence = text[start_index: start_index + maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(40):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = model.predict(x_pred, batch_size=1)[0,:]
                preds = preds[maxlen - 1, :]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
cb_logger = CSVLogger('model_fname2.log')

model.fit(x, y,
          batch_size=128,
          epochs=600,
          callbacks=[print_callback, cb_logger])

