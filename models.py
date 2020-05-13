import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.02
session = tf.Session(config=config)
KTF.set_session(session)

import pandas as pd
import numpy as np
import tqdm
import keras
from keras_gcn import GraphConv

embedding_layer = keras.layers.Embedding(input_dim=3335, output_dim=256, name='Embedding_layer')
f_data = keras.layers.Input(shape=(422291,), name='facebook_data')
f_data_e = embedding_layer(f_data)
f_edge = keras.layers.Input(shape=(422291, 422291), name='facebook_edge')

t_data = keras.layers.Input(shape=(669198,), name='twitter_data')
t_data_e = embedding_layer(t_data)
t_edge = keras.layers.Input(shape=(669198, 669198), name='twitter_edge')

f_conv_layer = GraphConv(
    units=32, step_num=1, name='GCN_for_facebook'
)([f_data_e, f_edge])
t_conv_layer = GraphConv(
    units=32, step_num=1, name='GCN_for_twitter'
)([t_data_e, t_edge])
y1 = keras.layers.Dense(1, activation='relu')(f_conv_layer)
y2 = keras.layers.Dense(1, activation='relu')(t_conv_layer)
y3 = keras.layers.Dot(axes=[-1, -1], name='transfor_matrix')([y1, y2])
y3=keras.layers.Activation(activation='sigmoid')(y3)
model = keras.models.Model([f_data, f_edge, t_data, t_edge], y3)
model.summary()

alignment = open('data/ASNet/alignment.csv').readlines()
facebook_networks = pd.read_csv('data/ASNet/facebook_network.csv')
facebook_username = open('data/ASNet/facebook_username.csv').readlines()
twitter_networks = pd.read_csv('data/ASNet/twitter_network.csv')
twitter_username = open('data/ASNet/twitter_username.csv').readlines()

f_user_dict = {}
t_user_dict = {}
charset = set()
for line in facebook_username:
    user_id = int(line.strip("\n").split(',')[0])
    user_name = line.strip("\n").split(',')[1]
    temp = user_name.strip(';').split(';')
    for i in temp:
        charset.add(i)
    if user_id not in f_user_dict.keys():
        f_user_dict[user_id] = [int(i) for i in temp]
    else:
        raise ValueError

for line in twitter_username:
    user_id = int(line.strip("\n").split(',')[0])
    user_name = line.strip("\n").split(',')[1]
    temp = user_name.strip(';').split(';')
    for i in temp:
        charset.add(i)
    if user_id not in t_user_dict.keys():
        t_user_dict[user_id] = [int(i) for i in temp]
    else:
        raise ValueError
data = []
for i in tqdm.tqdm(range(422291)):
    data.append(f_user_dict[i])
f_data = np.array(data)

data = []
for i in tqdm.tqdm(range(669198)):
    data.append(t_user_dict[i])
t_data = np.array(data)

del t_user_dict, f_user_dict, facebook_username, twitter_username, data

f_edge = np.zeros(shape=(422291, 422291))
for idx in tqdm.tqdm(facebook_networks.index):
    user_a = int(facebook_networks.loc[idx, 'facebook_id_a'])
    user_b = int(facebook_networks.loc[idx, 'facebook_id_b'])
    f_edge[user_a, user_a] = 1
    f_edge[user_a, user_b] = 1

t_edge = np.zeros(shape=(669198, 669198))
for idx in tqdm.tqdm(twitter_networks.index):
    user_a = int(twitter_networks.loc[idx, 'twitter_id_a'])
    user_b = int(twitter_networks.loc[idx, 'twitter_id_b'])
    t_edge[user_a, user_a] = 1
    t_edge[user_a, user_b] = 1

del twitter_networks, facebook_networks

labels = np.zeros(shape=(422291, 669198))
for line in tqdm.tqdm(alignment):
    f_u, t_u = line.strip('\n').split(',')
    labels[int(f_u), int(t_u)] = 1
del alignment
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
model.fit({'facebook_data':f_data,'facebook_edge':f_edge,
           'twitter_data':t_data,'twitter_edge':t_edge},labels,epochs=3,verbose=1,
          initial_epoch=0)
model.save('graph.h5')