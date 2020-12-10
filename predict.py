import pandas as pd
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow_hub as hub
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from sklearn import preprocessing
import keras
from keras.models import model_from_json
from keras.layers import Input, Lambda, Dense
from keras.models import Model,load_model
import keras.backend as K

df = pd.read_excel('train.xlsx')
df = df.drop('Unnamed: 1', axis=1)

df.messages = df.messages.apply(lambda x : x.replace('BOS',''))
df.messages = df.messages.apply(lambda x : x.replace('EOS',''))

elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)


x = list(df['messages'])
y = list(df['intents'])


le = preprocessing.LabelEncoder()

le.fit(y)

def encode(le,labels):
    enc = le.transform(labels)
    return keras.utils.to_categorical(enc)

def decode(le,one_hot):
    dec = np.argmax(one_hot,axis=1)
    return le.inverse_transform(dec)

x_enc = x
y_enc = encode(le,y)
x_train = np.asarray(x_enc[:3850])
y_train = np.asarray(y_enc[:3850])
x_test = np.asarray(x_enc[3850:])
y_test = np.asarray(y_enc[3850:])

def ELMoEmbedding(x):
    return elmo(tf.squeeze(tf.cast(x, tf.string)), signature="default" ,as_dict = True)["default"]


input_text = Input(shape=(1,) ,dtype=tf.string)
embedding = Lambda(ELMoEmbedding,output_shape=(1024,))(input_text)
dense = Dense(256, activation='relu')(embedding)
pred = Dense(151, activation='softmax')(dense)
model = Model(inputs =[input_text],outputs =pred)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


testing = 'Thanks for your response'

testing_=[]
testing_.append(testing)

testing_.append('')
pred_test = np.asarray(testing_)

with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    model.load_weights('./elmo_intent_model.h5')
    #predicts = model.predict(x_test,batch_size=32)
    predicts = model.predict(pred_test)

y_preds = decode(le,predicts)

print("The Intent is : ",y_preds[0])

