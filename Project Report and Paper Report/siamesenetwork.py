import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import keras
import keras.backend as K
from keras.layers import *
from keras.callbacks import *
from keras.optimizers import *
from keras.layers.core import Lambda, Flatten, Dense
from keras.layers import Bidirectional, LSTM
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Layer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

import pickle    
import os

train = pd.read_csv('data/Dataset.csv')
train=train[['text','class']]


module_url = 'https://tfhub.dev/google/universal-sentence-encoder-large/4'
# Import the Universal Sentence Encoder's TF Hub module
embed = hub.load(module_url)

input_text1 = Input(shape=(512,))
x = Dense(256, activation='relu')(input_text1)
x = Dropout(0.4)(x)
x = BatchNormalization()(x)
x = Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = Dropout(0.4)(x)
dense_layer = Dense(128, name='dense_layer')(x)
norm_layer = Lambda(lambda  x: K.l2_normalize(x, axis=1), name='norm_layer')(dense_layer)

model=Model(inputs=[input_text1], outputs=norm_layer)
model.summary()

#Input in batch of three
in_a = Input(shape=(512,))
in_p = Input(shape=(512,))
in_n = Input(shape=(512,))

#ouput embedding in batch of three
emb_a = model(in_a)
emb_p = model(in_p)
emb_n = model(in_n)

class TLL(Layer): #triplet Loss layer
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TLL, self).__init__(**kwargs)
    
    def triplet_loss(self, inputs):
        a, p, n = inputs
        p_dist = K.sum(K.square(a-p), axis=-1)
        n_dist = K.sum(K.square(a-n), axis=-1)

        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)
    
    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)

        return loss

def getTriples(unique_train_label,map_train_label_indices):
      label_l, label_r = np.random.choice(unique_train_label, 2, replace=False)
      a, p = np.random.choice(map_train_label_indices[label_l],2, replace=False)
      n = np.random.choice(map_train_label_indices[label_r])

      return a, p, n

def get_batch(k,train_set,unique_train_label,map_train_label_indices,embed):

    while True:
      idxs_a, idxs_p, idxs_n = [], [], []
      for _ in range(k):
          a, p, n = getTriples(unique_train_label,map_train_label_indices)
          idxs_a.append(a)
          idxs_p.append(p)
          idxs_n.append(n)
      a=train_set.iloc[idxs_a].values.tolist()
      b=train_set.iloc[idxs_p].values.tolist()
      c=train_set.iloc[idxs_n].values.tolist()
      a = embed(a)
      p = embed(b)
      n = embed(c)

      yield [a,p,n], []

tll = TLL(alpha=0.4, name='tll')([emb_a, emb_p, emb_n]) #layer that computes triplet loss
nnet_train = Model([in_a, in_p, in_n], tll) #model to be trained

unique_train_label=np.array(train['class'].unique().tolist())
labels_train=np.array(train['class'].tolist())
map_train_label_indices = {label: np.flatnonzero(labels_train == label) for label in unique_train_label}

nnet_train.compile(loss=None, optimizer='adam')
nnet_train.fit(get_batch(128,train['text'],unique_train_label,map_train_label_indices,embed), epochs=1,steps_per_epoch=10)

#experiment 01 (tweets vs articles)

df = pd.read_csv('data/Dataset.csv')
train = df[df['type']=='tweets']
test = df[df['type'] == 'NewsArticles']
pdf = pickle.load(open("data/Dataset.pkl","rb"))
tvec = pdf[pdf['type']=='tweets']['fvec'].values.tolist()
tevec = pdf[pdf['type']=='NewsArticles']['fvec'].values.tolist()

Xtr = model.predict(embed(np.array(train['text'].values.tolist()))) #obtaining embeddings
Xte = model.predict(embed(np.array(test['text'].values.tolist())))

#adding feature vector 
X_train = np.array([np.array(list(i)+j) for i,j in zip(tvec,Xtr)])
X_test = np.array([np.array(list(i)+j) for i,j in zip(tevec,Xte)])

y_train = np.array(train['class'].values.tolist())
y_test = np.array(test['class'].values.tolist())

svc = LinearSVC()
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f'Accuracy = {acc}')

#Experiment 02: Over the subjects (domains)
accScore = list()

for sub in ['COVID-19', 'General', 'Politics', 'SyriaWar']: #iterating over different domains

    train = df[df['subject']!=sub]
    test = df[df['subject'] == sub]
    tvec = pdf[pdf['subject']!=sub]['fvec'].values.tolist()
    tevec = pdf[pdf['subject'] == sub]['fvec'].values.tolist()

    Xtr = model.predict(embed(np.array(train['text'].values.tolist()))) #obtaining embeddings
    Xte = model.predict(embed(np.array(test['text'].values.tolist())))

    #adding feature vector 
    X_train = np.array([np.array(list(i)+j) for i,j in zip(tvec,Xtr)])
    X_test = np.array([np.array(list(i)+j) for i,j in zip(tevec,Xte)])

    y_train = np.array(train['class'].values.tolist())
    y_test = np.array(test['class'].values.tolist())

    svc = LinearSVC()
    svc.fit(X_train, y_train)

    y_pred = svc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    accScore.append(acc)

temp = round(sum(accScore)/len(accScore), 2)
print(f'Accuracy = {temp}')