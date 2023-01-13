import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import data_loader_special
import random
import math
import get_parameters
import pickle
import gzip
import scipy
from scipy.optimize import least_squares

# Only log errors (to prevent unnecessary cluttering of the console)
tf.logging.set_verbosity(tf.logging.ERROR)

class Config:
    def __init__(self):
        self.lattice_size = 8
        self.learning_rate = 1e-4
        self.batch_size = 128
        self.n_z = 6 # no of temp variables(generally 1)
        self.n_zrand = 2*6 # no of noise variables
        self.loss_type = 'log_gaussian' 
        self.datapoints = 320000
        self.n_temps = 32
        self.T_vals = np.linspace(0.05, 2.05, self.n_temps)
        self.is_train = True

class DeepLearningModel:
    def __init__(self, config):
        self.config = config
        self.x = tf.compat.v1.placeholder(name='x', dtype=tf.float32, shape=[None, config.lattice_size, config.lattice_size, 1])
        self.y = tf.compat.v1.placeholder(name='y', dtype=tf.float32, shape=[None, config.lattice_size, config.lattice_size, 1])
        self.temp = tf.compat.v1.placeholder(name='temp', dtype=tf.float32, shape=[None, config.n_z])
        self.create_encoder()
        self.create_decoder()

    def create_encoder(self):
        with tf.variable_scope('Encoder'):
            encoder_input = tf.concat([self.x, self.y], axis=-1)
            conv1 = tf.contrib.layers.conv2d(inputs=encoder_input, num_outputs=16, kernel_size=[3, 3], stride=1, padding='VALID', activation_fn=tf.nn.relu)
            conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=24, kernel_size=[3, 3], stride=1, padding='VALID', activation_fn=tf.nn.relu)
            conv3 = tf.contrib.layers.conv2d(inputs=conv2, num_outputs=32, kernel_size=[3, 3], stride=1, padding='VALID', activation_fn=tf.nn.relu)
            flat = tf.reshape(conv3, [-1, 2*2*32])
            net   = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=32, activation_fn=tf.tanh,scope='FC')
            net1  = tf.contrib.layers.fully_connected(inputs=net , num_outputs=10, activation_fn=tf.tanh,scope='hid')
            Z_mu  = tf.contrib.layers.fully_connected(inputs=net1, num_outputs=self.config.n_zrand,activation_fn=None,scope="mu")
            Z_sg  = tf.contrib.layers.fully_connected(inputs=net1, num_outputs=self.config.n_zrand,activation_fn=None,scope="sg")
            eps = tf.random_normal(shape=tf.shape(Z_mu),mean=0, stddev=1.0, dtype=tf.float32)
            z   = Z_mu + tf.sqrt(tf.exp(Z_sg)) * eps
    
    def create_decoder(self):
        with tf.variable_scope('Decoder'):
            net   = tf.contrib.layers.fully_connected(inputs= tf.concat([temp,z],axis=1),num_outputs=40, activation_fn=tf.tanh,scope="inp")
            net1  = tf.contrib.layers.fully_connected(inputs = net                      ,num_outputs=128,activation_fn=tf.tanh,scope="hid1")
            Conv  = tf.reshape(net1, [-1,4,4,8])
            Conv1 = tf.contrib.layers.conv2d_transpose(inputs=Conv, num_outputs=10, kernel_size=[3,3], stride=1, padding='VALID', activation_fn=tf.tanh, scope='Conv1') #(?,1,2,2) ->(?,6,6,1)
            Convmu= tf.contrib.layers.conv2d_transpose(inputs=Conv1,num_outputs=1 , kernel_size=[3,3], stride=1, padding='VALID', activation_fn=None, scope='Conv2mu') #(?,1,2,2) ->(?,6,6,1)
            Convsg= tf.contrib.layers.conv2d_transpose(inputs=Conv1,num_outputs=1 , kernel_size=[3,3], stride=1, padding='VALID', activation_fn=tf.nn.relu, scope='Conv2sg') #(?,1,2,2) ->(?,6,6,1)
            epsilon = tf.random_normal(shape = tf.shape(Convmu),mean =0.0 ,stddev = 1.0, dtype = tf.float32 )
            sample = Convmu + epsilon*tf.sqrt(tf.exp(-4*Convsg))#tf.random_normal(shape = tf.shape(net_mu),mean =net_mu ,stddev = tf.sqrt(tf.exp(-net_sg)), dtype = tf.float32 )#
    def set_lossfunction(self):
        with tf.variable_scope('loss'):
            # calculate reconstruction loss
            recon_loss = tf.reduce_mean(tf.square(self.x - self.Convmu))

            # calculate KL divergence loss
            KL_loss = 0.5 * tf.reduce_sum(tf.exp(self.Z_sg) + tf.square(self.Z_mu) - 1 - self.Z_sg, axis=1)
            KL_loss = tf.reduce_mean(KL_loss)

            # calculate total loss
            self.total_loss = recon_loss + KL_loss

            # define optimizer and training operation
            optimizer = tf.optimizers.Adam(learning_rate=config.learning_rate)
            self.train_op = optimizer.minimize(self.total_loss)

def return_intersection(hist_1, hist_2):# For evaluation : Calculates the % Overlap between two Histograms
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection
def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

config = Config()
model = DeepLearningModel(config)

total_loss = recon_loss + KL_loss
optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
train_op = optimizer.minimize(total_loss)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


saver = tf.compat.v1.train.Saver()
with tf.compat.v1.Session() as sess:
    if is_train == False:
        saver.restore(sess,'./CVAE_baselinexy2.ckpt')
    if is_train == True:
        training_data = data_loader_special.load_data_wrapper() #uploading the data
        tvals = np.repeat(T_vals,10000)
        c = list(zip(training_data,tvals))
        random.shuffle(c) # pairing and shuffling the data and temperature
        training_data, tvals = zip(*c)
        print(len(training_data),len(tvals))

        m = tf.compat.v1.placeholder(tf.float32,[datapoints, lattice_size, lattice_size,1])
        n = tf.compat.v1.placeholder(tf.float32,[datapoints, lattice_size, lattice_size,1])
        b = tf.compat.v1.placeholder(tf.float32,[datapoints, n_z])
        # Uploading the data to prevent memory issues prefetch and batching
        dataset = tf.data.Dataset.from_tensor_slices((m,n,b))
        dataset = dataset.prefetch(buffer_size=100)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        next = iterator.get_next()

        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(iterator.initializer,feed_dict = {m:training_data, b:np.repeat(np.array(tvals),n_z).reshape(datapoints,n_z), n:np.repeat(np.array(tvals),(lattice_size)**2).reshape(datapoints,lattice_size,lattice_size,1)})
        print("Session initialized :)")
        print("Iterator initialized :)")

        for i in range(30000):
            if i>0 and i % (datapoints // batch_size) == 0:
                sess.run(iterator.initializer, feed_dict = {m:training_data, b:np.repeat(np.array(tvals),n_z).reshape(datapoints,n_z), n:np.repeat(np.array(tvals),(lattice_size)**2).reshape(datapoints,lattice_size,lattice_size,1) })
            g,h,j = sess.run(next)
            _, Losses = sess.run([train_op, losses],feed_dict={x: g,y:h, temp:j })
            if i%1000==0:
                print(Losses)
        save_path = saver.save(sess, "./CVAE_baselinexy2.ckpt")
        print("Model saved in path: %s" % save_path)            