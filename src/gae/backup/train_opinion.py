from __future__ import division
from __future__ import print_function

import time
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from gae.optimizer import OptimizerAE, OptimizerVAE
from gae.input_data import load_data, load_data_1
from gae.model import GCNModelAE, GCNModelVAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edge_opinion

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 600, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 4, 'Number of units in hidden layer 2, P in our paper.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_vae', 'Model string.')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 0, 'Whether to use features (1) or not (0).')

flags.DEFINE_float('alpha_0', 3, 'prior of Beta distribution.')
flags.DEFINE_float('beta_0', 6.9, 'prior of Beta distribution.')
flags.DEFINE_float('p_encode', 0.05, 'trade off parameter of auto_encode.')
flags.DEFINE_float('p_kl', 0.01, 'trade off parameter of KL-divergence.')
flags.DEFINE_integer('KL_m', 10, 'approximate of the infinite sum in KLD.')
flags.DEFINE_float('test_rat', 0.9, 'test number of dataset.')
flags.DEFINE_integer('vae_epoch', 0, 'when add the auto-encoder loss and KL loss.')
flags.DEFINE_float('noise', 0.1, 'noise of synthetic data.')

model_str = FLAGS.model
dataset_str = FLAGS.dataset

seed = 123123
np.random.seed(seed)
tf.set_random_seed(seed)
# Load data
# adj, features = load_data(dataset_str)
adj, features = load_data_1()
# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

# adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
# adj = adj_train
adj_train = adj
y_train_belief, y_test_belief, y_train_un, y_test_un, train_mask, test_mask, omega_test = mask_test_edge_opinion(FLAGS.test_rat, FLAGS.noise)

if FLAGS.features == 0:
    features = sp.identity(y_test_belief.shape[0])  # featureless

# Some preprocessing
adj_norm = preprocess_graph(adj)

# Define placeholders
placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'labels_b': tf.placeholder(tf.float32, shape=(None, y_train_belief.shape[1])),
    'labels_un': tf.placeholder(tf.float32, shape=(None, y_train_un.shape[1])),
    'omega_test': tf.placeholder(tf.float32, shape=(None, y_train_belief.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'omega_t': tf.placeholder(tf.float32, shape=(None, omega_test.shape[1])),
}

num_nodes = adj.shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create model
model = None
if model_str == 'gcn_ae':
    model = GCNModelAE(placeholders, num_features, features_nonzero)
elif model_str == 'gcn_vae':
    model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

# Optimizer
with tf.name_scope('optimizer'):
    if model_str == 'gcn_ae':
        opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                      validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm)
    elif model_str == 'gcn_vae':
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes, pos_weight=pos_weight, norm=norm,
                           label_b=placeholders['labels_b'], label_un=placeholders['labels_un'],
                           mask=placeholders['labels_mask'], omega_t=placeholders['omega_t'],)

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_val = []
acc_val = []


def get_roc_score(edges_pos, edges_neg, emb=None):
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


cost_val = []
acc_val = []
val_roc_score = []

adj_label = adj_train + sp.eye(adj_train.shape[0])
# adj_label = adj_train
adj_label = sparse_to_tuple(adj_label)

# Train model
for epoch in range(FLAGS.epochs):
    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders, y_train_belief, y_train_un, train_mask, omega_test)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Run single weight update
    if epoch < FLAGS.vae_epoch:
        outs = sess.run([opt.opt_op, opt.cost, model.belief, model.uncertain], feed_dict=feed_dict)
    else:
        # ooooo = sess.run([model.r, model.s, model.alpha, model.beta, model.belief, model.uncertain, model.uni, model.z_n, model.z, model.reconstructions, opt.cost_decode], feed_dict=feed_dict)
        outs = sess.run([opt.opt_op1, opt.cost1, opt.cost_belief, opt.cost_uncertain, opt.cost_decode_sparse, opt.cost_decode_sparse], feed_dict=feed_dict)
    # Compute average loss
    avg_cost = outs[1]
    # avg_accuracy = outs[2]

    # roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false)
    # val_roc_score.append(roc_curr)

    # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
    #       "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
    #       "val_ap=", "{:.5f}".format(ap_curr),
    #       "time=", "{:.5f}".format(time.time() - t))
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost), "time=",
          "{:.5f}".format(time.time() - t))
    if np.mod(epoch + 1, 10) == 0:
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders, y_test_belief, y_test_un,
                                        test_mask, omega_test)
        feed_dict.update({placeholders['dropout']: 0.0})
        # Run single weight update
        outs = sess.run([opt.cost, opt.cost_belief, opt.cost_uncertain, opt.omega_mse], feed_dict=feed_dict)

        # Compute test loss
        test_cost = outs[1]
        print("mse_belief=", "{:.5f}".format(outs[1]), "mse_uncertain==", "{:.5f}".format(outs[2]), "bf-mse_opinion==", "{:.5f}".format(outs[3]))
print("Optimization Finished!")




# roc_score, ap_score = get_roc_score(test_edges, test_edges_false)
# print('Test ROC score: ' + str(roc_score))
# print('Test AP score: ' + str(ap_score))
