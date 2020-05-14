import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
session = tf.Session(config=config)
KTF.set_session(session)
import keras
from keras_gcn import GraphConv
import tqdm, json,numpy as np
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint,TensorBoard
from keras.losses import binary_crossentropy,categorical_crossentropy,cosine_proximity
from keras.preprocessing.sequence import pad_sequences

class Dataloader(object):
    def __init__(self,
                 batch_size=16, block_size=1000, split_rate=0.1,
                 alignment_path='data/ASNet/alignment.csv',
                 facebook_network_path='data/ASNet/facebook_network.csv',
                 facebook_username_path='data/ASNet/facebook_username.csv',
                 twitter_network_path='data/ASNet/twitter_network.csv',
                 twitter_username_path='data/ASNet/twitter_username.csv'
                 ):

        self.batch_size = batch_size
        self.block_size = block_size
        self.split_rate = split_rate
        self.max_len=256

        if not os.path.exists('data/ASNet/facebook_data.json') \
                or not os.path.exists('data/ASNet/twitter_data.json') \
                or not os.path.exists('data/ASNet/alignment.json'):
            self.alignment_path = alignment_path
            self.facebook_network_path = facebook_network_path
            self.facebook_username_path = facebook_username_path
            self.twitter_network_path = twitter_network_path
            self.twitter_username_path = twitter_username_path
            self.f_data, self.t_data, self.alignment,self.max_len = self.preprocess()
            print('max length:{}'.format(self.max_len))
        else:
            self.f_data = json.load(open('data/ASNet/facebook_data.json', 'r', encoding='utf-8'))
            self.t_data = json.load(open('data/ASNet/twitter_data.json', 'r', encoding='utf-8'))
            self.alignment = json.load(open('data/ASNet/alignment.json', 'r', encoding='utf-8'))
        self.train_t_keys, self.val_t_keys, self.train_f_keys, self.val_f_keys, \
        self.train_steps, self.val_steps = self.split_dataset(self.f_data, self.t_data)

    def preprocess(self):
        f_user_data = {}
        max_len=0
        f = open(self.facebook_username_path, 'r', encoding='utf-8').readlines()
        for line in tqdm.tqdm(f):
            user_id = int(line.strip("\n").split(',')[0])
            user_name = line.strip("\n").split(',')[1]
            temp = user_name.strip(';').split(';')
            if user_id not in f_user_data.keys():
                if len(temp)>max_len:max_len=len(temp)
                f_user_data[user_id] = {'user_name': temp, 'follows': []}
            else:
                raise ValueError
        del f

        f = open(self.facebook_network_path, 'r', encoding='utf-8').readlines()[1:]
        for line in tqdm.tqdm(f):
            user_a, user_b = line.strip('\n').split(',')
            f_user_data[int(user_a)]['follows'].append(int(user_b))

        del f

        t_user_data = {}
        f = open(self.twitter_username_path, 'r', encoding='utf-8').readlines()
        for line in tqdm.tqdm(f):
            user_id = int(line.strip("\n").split(',')[0])
            user_name = line.strip("\n").split(',')[1]
            temp = user_name.strip(';').split(';')
            if user_id not in t_user_data.keys():
                if len(temp)>max_len:max_len=len(temp)
                t_user_data[user_id] = {'user_name': temp, 'follows': []}
            else:
                raise ValueError
        del f

        f = open(self.twitter_network_path, 'r', encoding='utf-8').readlines()[1:]
        for line in tqdm.tqdm(f):
            user_a, user_b = line.strip('\n').split(',')
            t_user_data[int(user_a)]['follows'].append(int(user_b))

        del f

        f = open(self.alignment_path, 'r', encoding='utf-8').readlines()
        alignment = {}
        for line in tqdm.tqdm(f):
            f_u, t_u = line.strip('\n').split(',')
            if int(f_u) not in alignment.keys():
                alignment[int(f_u)] = [int(t_u)]
            else:
                alignment[int(f_u)].append(int(t_u))
        del f

        json.dump(f_user_data, open('data/ASNet/facebook_data.json', 'w', encoding='utf-8'))
        json.dump(t_user_data, open('data/ASNet/twitter_data.json', 'w', encoding='utf-8'))
        json.dump(alignment, open('data/ASNet/alignment.json', 'w', encoding='utf-8'))

        return f_user_data, t_user_data, alignment,max_len

    def split_dataset(self, f_data, t_data):
        num_f = len(f_data.keys())
        num_t = len(t_data.keys())
        val_f = int(self.split_rate * num_f)
        val_t = int(self.split_rate * num_t)
        train_f_keys = list(f_data.keys())[:(num_f - val_f)]
        val_f_keys = list(f_data.keys())[(num_f - val_f):]
        train_t_keys = list(t_data.keys())[:(num_t - val_t)]
        val_t_keys = list(t_data.keys())[(num_t - val_t):]
        train_steps = ((len(train_f_keys) // self.block_size + 1) // self.batch_size) * \
                      ((len(train_t_keys) // self.block_size + 1) // self.batch_size)
        val_steps = ((len(val_f_keys) // self.block_size + 1) // self.batch_size) * \
                    ((len(val_t_keys) // self.block_size + 1) // self.batch_size)
        return train_t_keys, val_t_keys, train_f_keys, val_f_keys, train_steps, val_steps

    def generator(self,is_train=True):
        f_keys=self.train_f_keys if is_train else self.val_f_keys
        t_keys=self.train_t_keys if is_train else self.val_t_keys
        f_keys=np.array(f_keys)
        t_keys=np.array(t_keys)
        while True:
            batch_f_edge=np.zeros(shape=(self.batch_size,self.block_size,self.block_size))
            batch_t_edge=np.zeros(shape=(self.batch_size,self.block_size,self.block_size))
            batch_labels=np.zeros(shape=(self.batch_size,self.block_size,self.block_size))
            f_data=np.zeros(shape=(self.batch_size,self.block_size,self.max_len))
            t_data=np.zeros(shape=(self.batch_size,self.block_size,self.max_len))
            for idx in range(self.batch_size):
                block_f_keys=np.random.choice(f_keys,size=self.block_size,replace=False)
                block_t_keys=np.random.choice(t_keys,size=self.block_size,replace=False)
                userid2idx_dict_f={}
                userid2idx_dict_t={}
                for i in range(self.block_size):
                    single_f_key=block_f_keys[i]
                    userid2idx_dict_f[single_f_key]=i
                    single_t_key=block_t_keys[i]
                    userid2idx_dict_t[single_t_key]=i
                    try:
                        f_user_name=self.f_data[str(single_f_key)]['user_name']
                        for t,j in enumerate(f_user_name):
                            f_data[idx,i,t]=int(j)
                        for t,j in enumerate(self.t_data[str(single_t_key)]['user_name']):
                            t_data[idx,i,t]=int(j)
                    except:
                        pass
                for i in range(self.block_size):
                    try:
                        single_f_key = block_f_keys[i]
                        followers_f = self.f_data[str(single_f_key)]['follows']
                        batch_f_edge[idx,i,i]=1
                        for follow in followers_f:
                            if int(follow) in list(block_f_keys):
                                batch_f_edge[idx,i,userid2idx_dict_f[int(follow)]]=1
                        single_t_key=block_t_keys[i]
                        followers_t=self.t_data[str(single_t_key)]['follows']
                        batch_t_edge[idx,i,i]=1
                        for follow in followers_t:
                            if int(follow) in list(block_t_keys):
                                batch_t_edge[idx,i,userid2idx_dict_t[int(follow)]]=1

                        if str(single_f_key) in self.alignment.keys():
                            aligs = self.alignment[str(single_f_key)]
                            for a in aligs:
                                if int(a) in list(block_t_keys):
                                    batch_labels[idx,i,userid2idx_dict_t[int(a)]]=1
                    except:
                        pass
            yield ({'facebook_data':f_data,'facebook_edge':batch_f_edge,
                   'twitter_data':t_data,'twitter_edge':batch_t_edge},{'prob_matrix':batch_labels})


class GraphNetwork(object):
    def __init__(self,batch_size=16,block_size=500):
        self.dataloader=Dataloader(batch_size=batch_size,block_size=block_size)

    def build_model(self):
        f_data = keras.layers.Input(shape=(None,self.dataloader.max_len), name='facebook_data')
        f_edge = keras.layers.Input(shape=(None, None), name='facebook_edge')

        t_data = keras.layers.Input(shape=(None,self.dataloader.max_len), name='twitter_data')
        t_edge = keras.layers.Input(shape=(None, None), name='twitter_edge')

        f_conv_layer = GraphConv(
            units=32, step_num=1, name='GCN_for_facebook'
        )([f_data, f_edge])
        t_conv_layer = GraphConv(
            units=32, step_num=1, name='GCN_for_twitter'
        )([t_data, t_edge])
        y1 = keras.layers.Dense(1, activation='relu')(f_conv_layer)
        y2 = keras.layers.Dense(1, activation='relu')(t_conv_layer)
        y3 = keras.layers.Dot(axes=[-1, -1], name='transfor_matrix')([y1, y2])
        y3 = keras.layers.Activation(activation='sigmoid',name='prob_matrix')(y3)
        model = keras.models.Model([f_data, f_edge, t_data, t_edge], y3)
        model.summary()
        return model

    def train(self):
        model=self.build_model()
        model.compile(
            optimizer=Adam(lr=1e-3),loss=binary_crossentropy,metrics=['acc']
        )
        os.makedirs('saved_models',exist_ok=True)
        model.fit_generator(
            generator=self.dataloader.generator(is_train=True),
            steps_per_epoch=500,
            validation_data=self.dataloader.generator(is_train=False),
            validation_steps=20,
            verbose=1,initial_epoch=0,epochs=100,callbacks=[
                TensorBoard('logs'),ReduceLROnPlateau(monitor='val_loss',patience=5,verbose=1),
                EarlyStopping(monitor='val_loss',patience=22,verbose=1),
                ModelCheckpoint('saved_models/graph_network.h5',verbose=1,monitor='val_loss',
                                save_best_only=True,save_weights_only=False,period=1)
            ]
        )

app=GraphNetwork(batch_size=4)
app.train()