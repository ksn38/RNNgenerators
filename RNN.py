import numpy as np
from keras.layers import Dense, LSTM, Input, concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, Callback
import os
from keras.models import Model
import pathlib

START_CHAR = '\b'
END_CHAR = '\t'
PADDING_CHAR = '\a'
chars = set([START_CHAR, '\n', END_CHAR])
with open('C:\\MLProjects\\venv7\\pasta.txt', 'r') as f:
    for line in f:
        chars.update(list(line.strip().lower()))#Возвращает копию указанной строки, с обоих концов которой устранены символы.
char_indices = {c: i for i, c in enumerate(sorted(list(chars)))}
char_indices[PADDING_CHAR] = 0
indices_to_chars = {i: c for c, i in char_indices.items()}
num_chars = len(chars)

def get_one(i, sz):
    res = np.zeros(sz)
    res[i] = 1
    return res

char_vectors = {
    c: (np.zeros(num_chars) if c == PADDING_CHAR else get_one(v, num_chars))
    for c, v in char_indices.items()
}

sentence_end_markers = set('.!?\n')
sentences = []
current_sentence = ''
with open('C:\\MLProjects\\venv7\\pasta.txt', 'r') as f:
    for line in f:
        s = line.strip().lower()
        if len(s) > 0:
            current_sentence += s + '\n'
        if len(s) == 0 or s[-1] in sentence_end_markers:
            current_sentence = current_sentence.strip()
            if len(current_sentence) > 10:
                sentences.append(current_sentence)
            current_sentence = ''

def get_matrices(sentences):
    max_sentence_len = np.max([len(x) for x in sentences])
    X = np.zeros((len(sentences), max_sentence_len, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), max_sentence_len, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        char_seq = (START_CHAR + sentence + END_CHAR).ljust(max_sentence_len + 1, PADDING_CHAR)
        for t in range(max_sentence_len):
            X[i, t, :] = char_vectors[char_seq[t]]
            y[i, t, :] = char_vectors[char_seq[t + 1]]
    return X, y

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

test_indices = np.random.choice(range(len(sentences)), int(len(sentences) * 0.05))
sentences_train = [sentences[x] for x in set(range(len(sentences))) - set(test_indices)]
sentences_test = [sentences[x] for x in test_indices]
sentences_train = sorted(sentences_train, key=lambda x: len(x))
X_test, y_test = get_matrices(sentences_test)
batch_size = 16

def generate_batch():
    while True:
        for i in range(int(len(sentences_train) / batch_size)):
            sentences_batch = sentences_train[i * batch_size: (i + 1) * batch_size]
            yield get_matrices(sentences_batch)

class CharSampler(Callback):
    def __init__(self, char_vectors, model):
        self.char_vectors = char_vectors
        self.model = model

    def on_train_begin(self, logs={}):
        self.epoch = 0
        if os.path.isfile("C:\\MLProjects\\venv7\\out4.txt"):
            os.remove("C:\\MLProjects\\venv7\\out4.txt")

    def sample(self, preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def sample_one(self, T):
        result = START_CHAR
        while len(result) < 500:
            Xsampled = np.zeros((1, len(result), num_chars))
            for t, c in enumerate(list(result)):
                Xsampled[0, t, :] = self.char_vectors[c]
            ysampled = self.model.predict(Xsampled, batch_size=1)[0, :]
            yv = ysampled[len(result) - 1, :]#выдает строку матрицы, нумерация которой len(result) - 1
            selected_char = indices_to_chars[self.sample(yv, T)]
            if selected_char == END_CHAR:
                break
            result = result + selected_char
        return result

    def on_epoch_end(self, batch, logs={}):
        self.epoch = self.epoch + 1
        if self.epoch % 30 == 0:
            print("\nEpoch %d text sampling:" % self.epoch)
            with open("C:\\MLProjects\\venv7\\out4.txt", 'a') as outf:
                outf.write('\n===== Epoch %d =====\n' % self.epoch)
                for T in [0.3, 0.5, 0.7, 0.9, 1.1]:
                    print('\tsampling, T = %.1f...' % T)
                    for _ in range(5):
                        self.model.reset_states()
                        res = self.sample_one(T)
                        outf.write('\nT = %.1f\n%s\n' % (T, res[1:]))

cb_sampler = CharSampler(char_vectors, model)
cb_logger = CSVLogger('model_fname.log')

model.fit_generator(generate_batch(), int(len(sentences_train) / batch_size) * batch_size, nb_epoch=500, verbose=True,
                    validation_data=(X_test, y_test), callbacks=[cb_logger, cb_sampler])
