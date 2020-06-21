#!/usr/bin/env python
# coding: utf-8

from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

import numpy as np

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
import sys, os
import numpy as np
import time
import csv
from sklearn.metrics import confusion_matrix
import scipy.io as sio
import random
import ABIDEParser as Reader

from keras.utils.np_utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import pickle as pkl 
import time
import copy
import scipy.spatial.distance
from tqdm import tqdm
from tensorflow.python.ops import array_ops



def glorot(shape, name=None):
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    var = tf.Variable(initial, name=name)
    return var
def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def dot(x, y, sparse=False):
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x,y)
    else:
        res = tf.matmul(x,y)
    return res

def accuracy(preds, labels):
    correct_prediction = tf.equal(tf.round(preds), labels)
    accuracy = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy)

def tens(shape, name=None):
    initial = tf.constant(10, tf.float32, shape)
    return tf.Variable(initial, name=name)



class gat_layer(object):
    def __init__(self, input_dim,F_, placeholders,attn_heads=1,attn_heads_reduction='concat',
                 activation=tf.nn.relu, use_bias=True,name_=''):
        self.dropout_rate = placeholders['dropout']
        self.in_drop = placeholders['in_drop']
        self.name = 'gat_layer'+name_
        self.vars = {}
        self.act = activation
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  #
        self.bias = use_bias
        self.A = placeholders["adj"]
        self.input_dim = input_dim

        with tf.variable_scope(self.name+'_vars'):
            for i in range(self.attn_heads):
                self.vars['weights_'+str(i)] = glorot([input_dim, F_], name='weights_' + str(i))
                self.vars["attn_self_weights_"+str(i)] = glorot([F_, 1], name='attn_self_weights_' + str(i))
                self.vars["attn_neighs_weights_"+str(i)] = glorot([F_, 1], name='attn_neighs_weights_' + str(i))
        if self.bias:
            self.vars['bias'] = zeros([F_],name='bias')
            
    def __call__(self, inputs):
        X = inputs
        if self.in_drop != 0.0:
            X = tf.nn.dropout(X, 1-self.in_drop)
        outputs = []
        dense_mask = []
        
        for head in range(self.attn_heads):
            # Compute inputs to attention network
            kernel = self.vars['weights_'+str(head)]
            features = tf.tensordot(X, kernel, axes=1)  # (N x F')

            # Compute feature combinations
            attention_self_kernel = self.vars["attn_self_weights_"+str(head)]
            attention_neighs_kernel = self.vars["attn_neighs_weights_"+str(head)]
            attn_for_self = tf.tensordot(features, attention_self_kernel, axes=1)    
            attn_for_neighs = tf.tensordot(features, attention_neighs_kernel, axes=1)  

            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            dense = attn_for_self + tf.transpose(attn_for_neighs, [0,2,1])  # (N x N) via broadcasting
            
            #print("plus:", dense.shape)

            # Add nonlinearty
            dense = tf.nn.leaky_relu(dense,alpha=0.2)

            zero_vec = -9e15*tf.ones_like(dense)
            dense = tf.where(self.A > 0.0, dense, zero_vec)
            dense_mask.append(dense)

            # Apply softmax to get attention coefficients
            dense = tf.nn.softmax(dense)  # (N x N)

            # Apply dropout to features and attention coefficients
            dropout_attn = tf.nn.dropout(dense, 1-self.dropout_rate) # (N x N)
            dropout_feat = tf.nn.dropout(features, 1-self.dropout_rate)  # (N x F')

            # Linear combination with neighbors' features
            node_features = tf.matmul(dropout_attn, dropout_feat)  # (N x F')

            if self.bias:
                node_features += self.vars["bias"]

            # Add output of attention head to final output
            if self.attn_heads_reduction == 'concat':
                outputs.append(self.act(node_features))
            else:
                outputs.append(node_features)

        # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = tf.concat(outputs, axis=-1)  # (N x KF')
        else:
            output = tf.add_n(outputs) / self.attn_heads  # N x F')
            output = self.act(output)

        return output, dense_mask




class fc_layer(object):

    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,sparse_input=False, act=tf.nn.relu, bias=False, featureless=False,name_=''):
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.name = 'fc_layer'+name_
        self.vars = {}
        self.act = act

        self.sparse_input = sparse_input
        self.featureless = featureless
        self.bias = bias

        with tf.variable_scope(self.name+'_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim], name='weights')
        if self.bias:
            self.vars['bias'] = zeros([output_dim],name='bias')

    def __call__(self, inputs):
        x = inputs

        x = tf.nn.dropout(x,1-self.dropout)

        output = tf.tensordot(x, self.vars['weights'], axes=1)

        if self.bias:
            output += self.vars['bias']
        return self.act(output)




class Model(object):

    def __init__(self, placeholders, input_dim):
        self.placeholders = placeholders
        self.input_dim = input_dim
        self.name = 'gat_mil'

        self.gat_layers = []
        self.fc_layers = []
        self.gcn_activations = []
        self.fc_activatinos = []

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.outputs = None
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]

        self.loss = 0
        self.accuracy = 0
        
        self.node_prob = None
        self.dense_mask = []

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.opt_op = None
        
        self.loss_explainer = 0
        self.optimizer_explainer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
        self.opt_op_explainer = None
        self.M = tens((FLAGS.node_num, FLAGS.node_num), name='mask')
        
        self.build()

    def build(self):
        with tf.variable_scope(self.name):
            self._build()

        sigmoid_M = tf.sigmoid(self.M)
        self.inputs = tf.multiply(self.inputs, sigmoid_M)

        self.gcn_activations.append(self.inputs)

        for layer in self.gat_layers:
            hidden, dense_mask = layer(self.gcn_activations[-1])
            self.gcn_activations.append(hidden) 
            self.dense_mask.append(dense_mask)


        p_layer = self.fc_layers[0]
        node_prob = p_layer(self.gcn_activations[-1])
        
        tensor = tf.reshape(node_prob, shape=(-1, FLAGS.node_num))
        layer = self.fc_layers[1]
        attention_prob = layer(tensor)


        attention_mul = tf.multiply(tensor, attention_prob)
        self.outputs = tf.reduce_sum(attention_mul, 1, keep_dims=True)
        print(self.outputs.shape)

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        self._loss()
        self._accuracy()


        var_list = tf.trainable_variables()
        var_list1 = []
        for var in var_list:
            if var != self.M:
                var_list1.append(var)
            elif var == self.M:
                #stop = input("M exit!!!!!!!")
                pass
        
        self.opt_op = self.optimizer.minimize(self.loss, var_list = [var_list1])
        self.loss_explainer += tf.reduce_mean(tf.losses.log_loss(labels=self.placeholders['labels'], predictions=self.outputs))
        self.opt_op_explainer = self.optimizer_explainer.minimize(self.loss_explainer, var_list=[self.M])

    def _build(self):
        self.gat_layers.append(gat_layer(input_dim=self.input_dim,F_=FLAGS.hidden1_gat, placeholders=self.placeholders,
                                         attn_heads=FLAGS.attn_heads,attn_heads_reduction='concat',
                                         activation=tf.nn.leaky_relu, use_bias=True,name_='1'))

        self.gat_layers.append(gat_layer(input_dim=FLAGS.hidden1_gat*FLAGS.attn_heads,F_=FLAGS.output_gat, placeholders=self.placeholders,
                                         attn_heads=3,attn_heads_reduction='average',
                                         activation=tf.nn.leaky_relu, use_bias=True,name_='2'))

        self.fc_layers.append(fc_layer(input_dim=FLAGS.output_gat, output_dim=FLAGS.output_dim, placeholders=self.placeholders, 
                                       act=tf.nn.sigmoid, dropout=True, name_='1'))
        
        self.fc_layers.append(fc_layer(input_dim=FLAGS.node_num, output_dim=FLAGS.node_num, placeholders=self.placeholders, 
                                       act=tf.nn.softmax, dropout=True, name_='2'))

    def _loss(self):
        for var in self.gat_layers[0].vars.values():
            self.loss += FLAGS.weight_decay*tf.nn.l2_loss(var)

            
        self.loss += tf.reduce_mean(tf.losses.log_loss(labels=self.placeholders['labels'], predictions=self.outputs))

    def _accuracy(self):
        self.accuracy = accuracy(self.outputs, self.placeholders['labels'])

    def predict(self):
        return self.outputs





flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('node_num', 110, 'Number of Graph nodes')

flags.DEFINE_integer('output_dim', 1, 'Number of output_dim')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate') #0.0005，0.0001，0.00005，0.00001，0.00003
flags.DEFINE_integer('batch_num', 10, 'Number of epochs to train')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train')
flags.DEFINE_integer('attn_heads', 5, 'Number of attention head')

flags.DEFINE_integer('hidden1_gat', 24, 'Number of units in hidden layer 1 of gcn')
flags.DEFINE_integer('output_gat', 3, 'Number of units in output layer 1 of gcn')

flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('in_drop', 0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 15, 'Tolerance for early stopping (# of epochs).')



root_folder = '/data/abide/ABIDE_pcp'
data_folder = os.path.join(root_folder, 'cpac/filt_noglobal')
subject_IDs = np.genfromtxt("/data/abide/tool/1035_subject_ids.txt", dtype=str)
subject_IDs = subject_IDs.tolist()


# Get subject labels
label_dict = Reader.get_label(subject_IDs)
label_list = np.array([int(label_dict[x]) for x in subject_IDs])





def load_connectivity(subject_list, kind, atlas_name = 'aal'):
    all_networks = []
    for subject in subject_list:
        fl = os.path.join(data_folder, subject,
                          subject + "_" + atlas_name + "_" + kind + ".mat")
        matrix = sio.loadmat(fl)['connectivity']
        if atlas_name == 'ho':
            matrix = np.delete(matrix, 82, axis=0)
            matrix = np.delete(matrix, 82, axis=1)
        all_networks.append(matrix)
    all_networks=np.array(all_networks)
    return all_networks



def getconn_vector(subject_name0, kind, atlas):
    subject_name = np.array(subject_name0)
    data_x = []
    data_y = []
    conn_array = load_connectivity(subject_name, kind, atlas)
    data_x = np.array(conn_array)
    
    for subname in subject_name:
        data_y.append([int(label_dict[subname])])
    
    data_y = np.array(data_y)
    return data_x, data_y




X, Y = getconn_vector(subject_IDs, "correlation", "ho")


abs_x = map(abs, X)
adjs = np.array(list(abs_x))
features = X



tf.app.flags.DEFINE_string('f', '', 'kernel')



# Define placeholders
placeholders = {
    'adj': tf.placeholder(tf.float32, shape=(None,FLAGS.node_num,FLAGS.node_num)),
    'features': tf.placeholder(tf.float32, shape=(None,FLAGS.node_num,FLAGS.node_num)),
    'labels': tf.placeholder(tf.float32, shape=(None, 1)),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'in_drop': tf.placeholder_with_default(0., shape=()),
}




def construct_feed_dict(features, support, labels, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: support})
    return feed_dict



# Define model evaluation function
def evaluate(features, support, labels, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)




def shuffle(adjs, features, y):
    shuffle_ix = np.random.permutation(np.arange(len(y)))
    adjs = adjs[shuffle_ix]
    features = features[shuffle_ix]
    y = y[shuffle_ix]
    return adjs, features, y

all_test_index = []
all_train_index = []
skf = StratifiedKFold(n_splits=5,shuffle=True, random_state=42)
for train_index, test_index in skf.split(subject_IDs, label_list):
    all_train_index.append(train_index)
    all_test_index.append(test_index)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import balanced_accuracy_score
results = []

l1, l2 = [], []

k = 4
train_index = all_train_index[k]
test_index = all_test_index[k]

bestModelSavePath0 = '/gat-nr-gnne/fold_e_mask%s/gat_e%s_weights_best.ckpt' % (str(k),str(k))
bestModelSavePath1 = '/gat-nr-gnne/fold_e_mask%s/gat_e%s_weights_best.ckpt' % (str(k),str(k))
features_train, features_test = features[train_index], features[test_index]
support_train, support_test = adjs[train_index], adjs[test_index]
y_train, y_test = Y[train_index], Y[test_index]
    
support_train, features_train, y_train = shuffle(support_train, features_train, y_train)
support_test, features_test, y_test = shuffle(support_test, features_test, y_test)
print("test shape:",features_test.shape, "support shape:", support_train.shape)
print("y_test", y_test[:3])

model = Model(placeholders, input_dim=features.shape[2])
sess = tf.Session()
sess.run(tf.global_variables_initializer())
'''设置模型保存器'''
m_saver = tf.train.Saver()

cost_val = []
batch_num = FLAGS.batch_num
train_num = features_train.shape[0]
for epoch in range(FLAGS.epochs):  #FLAGS.epochs
    batches = range(round(train_num / batch_num))
        
    costs = np.zeros((len(batches), 2))
    accs = np.zeros((len(batches), 2))

    t = time.time()

    for ib in batches:
        from_i = ib * batch_num
        to_i = (ib+1) * batch_num
            
        features_train_batch = features_train[from_i:to_i]
        support_train_batch = support_train[from_i:to_i]
        y_train_batch = y_train[from_i:to_i]
        feed_dict = construct_feed_dict(features_train_batch, support_train_batch, y_train_batch, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        feed_dict.update({placeholders['in_drop']: FLAGS.in_drop})

        outs = sess.run([model.opt_op, model.loss, model.accuracy],feed_dict=feed_dict)

        cost,acc,duration = evaluate(features_test, support_test, y_test, placeholders)
        costs[ib] = [outs[1], cost]
        accs[ib] = [outs[2], acc]
            
    costs = costs.mean(axis=0)
    cost_train, cost_test = costs
    cost_val.append(cost_test)
        
    accs = accs.mean(axis=0)
    acc_train, acc_test = accs
        
        
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cost_train),
        "train_acc=", "{:.5f}".format(acc_train), "test_loss=", "{:.5f}".format(cost_test),
        "test_acc=", "{:.5f}".format(acc_test), "time=", "{:.5f}".format(time.time() - t))
        
    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("early_stopping...")
        break
m_saver.save(sess, bestModelSavePath0)
print("Optimization Finished!")

test_cost, test_acc, test_duration = evaluate(features_test, support_test, y_test, placeholders)

print("Test set results:", "cost=", "{:.5f}".format(test_cost),
    "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
l1.append(test_acc)
test_pre = model.predict()
feed_dict = construct_feed_dict(features_test, support_test, y_test, placeholders)
test_pre = sess.run([test_pre],feed_dict=feed_dict)

y_pred = []
for p in test_pre[0]:
    y_pred.append(round(p[0]))
y_pred = np.array(y_pred)

[[TN, FP], [FN, TP]] = confusion_matrix(y_test, y_pred, labels=[0, 1]).astype(float)
acc = (TP+TN)/(TP+TN+FP+FN)
specificity = TN/(FP+TN)
sensitivity = recall = TP/(TP+FN)
fscore = f1_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test,test_pre[0])
roc_auc = auc(fpr, tpr)
print("accuracy, sensivity, specificity, fscore, auc:", acc, sensitivity, specificity, fscore, roc_auc)
result = [acc, sensitivity, specificity, fscore, roc_auc]
 
cost_val = []
for epoch in range(400):  #FLAGS.epochs
    batches = range(round(train_num / batch_num))
        
    costs = np.zeros((len(batches), 2))
    accs = np.zeros((len(batches), 2))

    t = time.time()

    for ib in batches:
        from_i = ib * batch_num
        to_i = (ib+1) * batch_num
            
        features_train_batch = features_train[from_i:to_i]
        support_train_batch = support_train[from_i:to_i]
        y_train_batch = y_train[from_i:to_i]
        feed_dict = construct_feed_dict(features_train_batch, support_train_batch, y_train_batch, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        feed_dict.update({placeholders['in_drop']: FLAGS.in_drop})

        outs = sess.run([model.opt_op_explainer, model.loss_explainer, model.accuracy],feed_dict=feed_dict)

        cost,acc,duration = evaluate(features_test, support_test, y_test, placeholders)
        costs[ib] = [outs[1], cost]
        accs[ib] = [outs[2], acc]
            
    costs = costs.mean(axis=0)
    cost_train, cost_test = costs
    cost_val.append(cost_test)
        
    accs = accs.mean(axis=0)
    acc_train, acc_test = accs
        
        
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cost_train),
        "train_acc=", "{:.5f}".format(acc_train), "test_loss=", "{:.5f}".format(cost_test),
        "test_acc=", "{:.5f}".format(acc_test), "time=", "{:.5f}".format(time.time() - t))
        

m_saver.save(sess, bestModelSavePath1)
print("Optimization Finished!")

test_cost, test_acc, test_duration = evaluate(features_test, support_test, y_test, placeholders)

print("Test set results:", "cost=", "{:.5f}".format(test_cost),
    "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
l2.append(test_acc)
    
M = model.M
s_m = tf.sigmoid(M)
M = M.eval(session=sess)
s_m = s_m.eval(session=sess)

with open("/weight/gnne/M"+str(k),'wb+') as f:
    pkl.dump(s_m, f) 
    
sess.close()


print(l1)
print(l2)

print(result)






