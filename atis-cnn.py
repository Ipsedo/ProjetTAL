#!/usr/bin/python
# coding: utf-8

import numpy as np
import json
from collections import Counter
from gensim.models import Word2Vec
import pandas as pd
import pickle
from keras import preprocessing

# 1. ne récupérer que la partie texte et intent des données
def readatis(filename='data/atis-2.train.w-intent.iob'):
    data = pd.read_csv(filename, sep='\t', header=None)
    sents = [s.split() for s in data[0].tolist()]
    ners  = [s.split() for s in data[1].tolist()]
    # remplacer les chiffres de sents par #
    for i, sent in enumerate(sents):
        sent = ' '.join(sent)
        for d in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            sent = sent.replace(d, '#')
        sents[i] = sent.split()
    # l'intent est le dernier élément de ners
    ints = [s[-1] for s in ners]
    assert(len(sents)==len(ints))
    return sents, ners, ints

# 2. on applique la préparation des données sur trn, dev et test
# on utilise tout all_atis.iob car ce programme ne supporte pas
# d'avoir des labels inconnus dans dev ou test
all_texts, all_ners, all_ints = readatis('data-atis/all_atis.iob')

# 3. on regarde combien d'intent pour avoir une idée
ll = len(list(set(all_ints)))
print('nb intents : ', ll)
print(Counter(all_ints).most_common(10))

# 4. apprendre le w2v
# 4.1 on retire les marqueurs de début et fin de sent ie BOS et EOS
w2v_text = [s[1:-1] for s in all_texts]

# 4.2 apprentissage et sauvegarde du modèle
model = Word2Vec(w2v_text, size=100, min_count=0, window=5, workers=3, iter=5)
model.save('models/atis_w2v.gensimmodel')
print(len(model.wv.vocab))
print('training done!')

# 4.3 on constitue le vocabulaire et on test la similarité sur un exemple
vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
t1 = model.wv.most_similar('plane')
t2 = model.wv.most_similar('flights')
t3 = model.wv.most_similar('baltimore')
t4 = model.wv.most_similar('arrive')
print('similaire à plane : ',t1,'\n')
print('similaire à flights : ',t2,'\n')
print('similaire à baltimore : ',t3,'\n')
print('similaire à arrive : ',t4,'\n')

# imprimer les mots les plus communs
print(model.wv.index2word[0], model.wv.index2word[1], model.wv.index2word[2])
# imprimer les mots les moins communs
vocab_size = len(model.wv.vocab)
print(model.wv.index2word[vocab_size - 1], model.wv.index2word[vocab_size - 2], model.wv.index2word[vocab_size - 3])
# imprimer l'index du mot 'plane'
print('Index of "plane" is: {}'.format(model.wv.vocab['plane'].index))

# # format dans des tenseurs
# from keras import preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# fixer un nb max de mots, ici 10000
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(w2v_text) #attention prendre ce qui ne contient ni eos ni bos
sequences = tokenizer.texts_to_sequences([s[1:-1] for s in all_texts])

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# de façon arbitraire une sequence aura 100 mots max
# data = pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0)
data = pad_sequences(sequences, maxlen=100)

# transformer les labels en index/entier
labels_index = {} #dict(nom de label, numéro)
labels_name = [] #tableau de nom indexé par les numéros

for l in all_ints:
    if not l in labels_index:
        label_id = len(labels_name)
        labels_index[l] = label_id
        labels_name.append(l)
        
labels = []
for l in all_ints:
    labels.append(labels_index[l])

from keras.utils import to_categorical
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# splitter les données pour obtenir données trn et dev
print(data.shape[0])
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(0.2 * data.shape[0])
print(nb_validation_samples)
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

# 5. préparation de la couche embedding
# 5.1 calculer le embedding matrix
embedding_matrix = np.zeros((len(model.wv.vocab) +1, 100))
for i in range(len(model.wv.vocab)):
    embedding_vector = model.wv[model.wv.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print('embedding matrix ok')

# 5.2 charger cette matrice dans le layer embedding
from keras.layers import Embedding
embedding_layer = Embedding(len(word_index) +1,
                            100,
                            weights=[embedding_matrix],
                            input_length=100,
                            trainable=False)
print('matrice chargée dans le layer embedding')

# 6. construction du réseau de convolution 1D
from keras.layers import Input
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Model

sequence_input = Input(shape=(100,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, padding='same', activation='relu')(embedded_sequences) #en same, pas de division, on garde 100
x = MaxPooling1D(5)(x) #divise 100/5=20
x = Conv1D(128, 5, padding='same', activation='relu')(x)
x = MaxPooling1D(20)(x)  # global max pooling on prend ce qui reste
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

# happy learning!
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=2, batch_size=128)
# model.fit(x_train, y_train, epochs=2, batch_size=128)

