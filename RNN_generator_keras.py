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

#pasta0 = []
path = get_file('C:/MLProjects/venv7/012.txt', origin='')
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
#print('corpus length:', len(text))
#pasta0.append('len(text)')
#pasta0.append(len(text))
#pasta0.append('\n')

chars = sorted(list(set(text)))
#print(chars)
#pasta0.append('chars:')
#pasta0.append(chars)
#pasta0.append('\n')
#print('total chars:', len(chars))
#pasta0.append('total chars:')
#pasta0.append(len(chars))
#pasta0.append('\n')
char_indices = dict((c, i) for i, c in enumerate(chars))
print(char_indices)
#pasta0.append('char_indices:')
#pasta0.append(char_indices)
#pasta0.append('\n')
#print('char_indices', char_indices)
indices_char = dict((i, c) for i, c in enumerate(chars))
print(indices_char)
#pasta0.append('indices_char:')
#pasta0.append(indices_char)
#pasta0.append('\n')
#print('indices_char', indices_char)

# cut the text in semi-redundant sequences of maxlen characters
maxlen =5
#pasta0.append('maxlen:')
#pasta0.append(maxlen)
#pasta0.append('\n')
step = 3
#pasta0.append('step:')
#pasta0.append(step)
#pasta0.append('\n')
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
# print('nb sequences:', len(sentences))

#pasta3 = []
print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    print('i --- ', i, type(i))
    print('sentence --- ', sentence, type(sentence))
    for t, char in enumerate(sentence):
        print('t --- ', t, type(t))
        print('char --- ', char, type(char))
        print('char_indices[char] --- ', char_indices[char], type(char_indices[char]))
        x[i, t, char_indices[char]] = 1
        #pasta0.append('x')
        #pasta0.append(x)
    y[i, char_indices[next_chars[i]]] = 1
    print('char_indices[next_chars[i]] --- ', char_indices[next_chars[i]])
    #pasta0.append('y')
    #pasta0.append(y)

# build the model: a single LSTM
# print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

o = open('out.txt', 'w')
o.write('')
#pasta0 = (''.join(map(str, pasta0)))
#o.write(pasta0)
o.close()
o = open('out2.txt', 'w')
o.write('')
o.close()
#o = open('out3.txt', 'w')
#o.write('')
#o.write(str(#pasta3))
#o.close()

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    ##pasta2 = []
    preds = np.asarray(preds).astype('float64')
    ##pasta2.append(preds)
    ##pasta2.append('\n')
    preds = np.log(preds) / temperature
    ##pasta2.append(preds)
    ##pasta2.append('\n')
    exp_preds = np.exp(preds)
    ##pasta2.append(exp_preds)
    ##pasta2.append('\n')
    preds = exp_preds / np.sum(exp_preds)
    ##pasta2.append(preds)
    ##pasta2.append('\n')
    probas = np.random.multinomial(1, preds, 1)
    ##pasta2.append(probas)
    ##pasta2.append('\n')
    ##pasta2 = (''.join(map(str, #pasta2)))
    #o = open('out2.txt', 'a')
    #o.write(str(#pasta2))
    #o.close()
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    # print()
    print('----- Generating text after Epoch: %d' % epoch)
    #pasta = []
    #pasta25 = []

    #pasta.append(epoch + 1)

    start_index = random.randint(0, len(text) - maxlen - 1)
    #pasta25.append(start_index)
    #pasta25.append('\n')

    for diversity in [0.9]:
        #pasta.append('\n diversity ')
        print('----- diversity:', diversity)
        #pasta.append(diversity)
        #pasta.append('\n sentence1 ')

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        # print('----- Generating with seed: "' + sentence + '"')

        sys.stdout.write(sentence)
        #pasta.append(generated)

        for i in range(9):
            #pasta.append('\n')
            #pasta.append('n:')
            #pasta.append(i+1)
            #pasta.append('\n')
            #pasta.append('x_pred')
            #Создает нулевой тензор с векторами one-hot без len(sentences) из конца потока существующих и рассчитанных данных
            x_pred = np.zeros((1, maxlen, len(chars)))
            #pasta.append(x_pred)
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.
                #на каждом шаге выдает maxlen тензоров со сдвигом на 1 строчку вперед
                #pasta25.append(x_pred)
                #pasta25.append(t+1)

            #сырые данные модели преобразуются в символ
            preds = model.predict(x_pred, verbose=0)[0]
            #pasta.append('\n')
            #pasta.append('preds ')
            #pasta.append(preds)
            #pasta.append('\n')
            next_index = sample(preds, diversity)
            #pasta.append('next_index ')
            #pasta.append(next_index)
            #pasta.append('\n')
            next_char = indices_char[next_index]
            #pasta.append('next_char ')
            #pasta.append(next_char)
            #pasta.append('\n')

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
            #pasta.append('sentence ')
            #pasta.append(sentence)


    #pasta = (''.join(map(str, #pasta)))
    #o = open('out.txt', 'a')
    #o.write(pasta)
    #o.write(str(#pasta25))
    #o.close()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y,
          batch_size=128,
          epochs=1,
          callbacks=[print_callback])