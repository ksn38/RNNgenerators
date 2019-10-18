from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

path = get_file('C:/MLProjects/venv7/pastaa.txt', origin='')
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
#print('corpus length:', len(text))

chars = sorted(list(set(text)))
#print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 4
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
#print('nb sequences:', len(sentences))

#print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
#print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

o = open('out.txt', 'w')
o.write('')
o.close()
o = open('out2.txt', 'w')
o.write('')
o.close()

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    #print()
    #print('----- Generating text after Epoch: %d' % epoch)
    pasta = []
    pasta2 = []
    pasta.append(epoch + 1)
    pasta.append('\n')

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        #print('----- diversity:', diversity)
        pasta.append(diversity)
        pasta.append('\n')

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        #print('----- Generating with seed: "' + sentence + '"')

        sys.stdout.write(sentence)
        pasta.append(generated)
        pasta.append('\n')

        for i in range(200):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
            #print()
            pasta.append(sentence)
            pasta.append('\n')
            pasta2.append(generated)
            pasta2.append('\n')
            #pasta2 = (''.join(map(str, pasta2)))
            #o = open('out2.txt', 'a')
            #o.write(str(pasta2))
    pasta = (''.join(map(str, pasta)))
    o = open('out.txt', 'a')
    o.write(pasta)


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y,
          batch_size=128,
          epochs=2,
          callbacks=[print_callback])