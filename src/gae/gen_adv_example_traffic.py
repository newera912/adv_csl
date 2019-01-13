from __future__ import division
from __future__ import print_function

import time
import os
import json
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from decimal import Decimal

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from optimizer import OptimizerAE, OptimizerVAE
from input_data import load_data_epinion
from model import GCNModelAE, GCNModelVAE
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edge_epinion

import pickle,copy


def clip01(x):
    if x<0.0: return 0.0
    elif x>1.0: return 1.0
    else: return x

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 30, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 4, 'Number of units in hidden layer 2, P in our paper.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_vae', 'Model string.')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 0, 'Whether to use features (1) or not (0).')

# flags.DEFINE_float('alpha_0', 3, 'prior of Beta distribution.')
# flags.DEFINE_float('beta_0', 6.9, 'prior of Beta distribution.')
flags.DEFINE_float('p_encode', 0.01, 'trade off parameter of auto_encode.')
flags.DEFINE_float('p_kl', 0.01, 'trade off parameter of KL-divergence.')
flags.DEFINE_integer('KL_m', 10, 'approximate of the infinite sum in KLD.')
flags.DEFINE_float('test_rat', 0.2, 'test number of dataset.')
flags.DEFINE_integer('vae_epoch', 0, 'when add the auto-encoder loss and KL loss.')
flags.DEFINE_float('ref_speed', 0.8, 'noise of synthetic data.')
flags.DEFINE_integer('T', 10, 'time winidow.')

model_str = FLAGS.model
dataset_str = FLAGS.dataset

seed = 123123
np.random.seed(seed)
tf.set_random_seed(seed)
# Load data
# adj, features = load_data(dataset_str)
data_root = "/network/rit/lab/ceashpc/adil/data/adv_csl/Jan2/traffic/random_pgd_gcn_vae/"
org_data_root="/network/rit/lab/ceashpc/adil/data/adv_csl/Jan2/traffic/random_pgd/"
#real data
case_count=0
realizations=10
alpha = 0.01
ref_pers = [0.6,0.7, 0.8]
datasets = ['philly', 'dc']
T=43
for adv_type in ["random_flip","random_noise","random_pgd","random_pgd_gcn_vae"][3:]:
    for real_i in range(realizations)[:]:
        for ref_per in ref_pers[:1]:
            for dataset in datasets[1:]:
                out_folder = data_root + dataset + "/"
                if not os.path.exists(out_folder):
                    os.makedirs(out_folder)
                for weekday in range(5)[:1]:
                    for hour in range(8, 22)[:1]:
                        for test_ratio in [0.1, 0.2, 0.3, 0.4, 0.5][:]:
                            fileName = org_data_root +dataset+ '/network_{}_weekday_{}_hour_{}_refspeed_{}-testratio-{}-gamma-{}-realization-{}.pkl'.format(
                                dataset, weekday, hour, ref_per, test_ratio, 0.0, real_i)

                            adj,y_train_belief, y_test_belief, y_train_un, y_test_un, train_mask, test_mask, omega_test, alpha_0, beta_0 = mask_test_edge_epinion(fileName,T)
                            print(len(y_test_belief),len(y_train_belief),len(train_mask),len(test_mask))
                            count=0
                            for a in test_mask:
                                if a==True:
                                    count+=1
                            print("Test nodes:",count)
                            # Store original adjacency matrix (without diagonal entries) for later
                            adj_orig = adj
                            adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
                            adj_orig.eliminate_zeros()

                            # adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
                            # adj = adj_train
                            adj_train = adj
                            # y_train_belief, y_test_belief, y_train_un, y_test_un, train_mask, test_mask, omega_test, alpha_0, beta_0 = mask_test_edge_epinion(FLAGS.test_rat, T=FLAGS.T)

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
                                'alpha_0': tf.placeholder(tf.float32),
                                'beta_0': tf.placeholder(tf.float32)
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


                            adj_label = adj_train + sp.eye(adj_train.shape[0])
                            # adj_label = adj_train
                            adj_label = sparse_to_tuple(adj_label)


                            bf_mse = []
                            b_mse = []
                            u_mse = []
                            for k in range(1):
                                # Train model
                                sess.run(tf.global_variables_initializer())
                                _,y_train_belief, y_test_belief, y_train_un, y_test_un, train_mask, test_mask, omega_test, alpha_0, beta_0 = mask_test_edge_epinion(fileName, T)
                                bf_mse_one = []
                                b_mse_one = []
                                u_mse_one = []
                                belief_one=[]
                                uncertain_one=[]
                                gradients = []
                                t1 = time.time()
                                for epoch in range(FLAGS.epochs):
                                    t = time.time()
                                    # Construct feed dictionary
                                    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders, y_train_belief, y_train_un,
                                                                    train_mask, omega_test, alpha_0, beta_0)
                                    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                                    # Run single weight update
                                    if epoch < FLAGS.vae_epoch:
                                        outs = sess.run([opt.opt_op, opt.cost, model.belief, model.uncertain], feed_dict=feed_dict)
                                    else:
                                        # ooooo = sess.run([model.r, model.s, model.alpha, model.beta, model.belief, model.uncertain, model.uni, model.z_n, model.z, model.reconstructions, opt.cost_decode], feed_dict=feed_dict)
                                        outs = sess.run([opt.opt_op1, opt.cost1, opt.cost_belief, opt.cost_uncertain, opt.cost_decode_sparse,
                                                         opt.cost_decode_sparse,opt.grads_vars], feed_dict=feed_dict)
                                    # Compute average loss
                                    avg_cost = outs[1]
                                    gradients.append(outs[6])

                                    if np.mod(epoch + 1, 10) == 0:
                                        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders, y_test_belief, y_test_un,
                                                                        test_mask, omega_test, alpha_0, beta_0)
                                        feed_dict.update({placeholders['dropout']: 0.0})
                                        # Run single weight update
                                        outs = sess.run([opt.cost, opt.cost_belief, opt.cost_uncertain, opt.omega_mse,opt.belief,opt.uncertain], feed_dict=feed_dict)

                                        # Compute test loss
                                        test_cost = outs[1]
                                        print("mse_belief=", "{:.5f}".format(outs[1]), "mse_uncertain==", "{:.5f}".format(outs[2]),
                                              "bf-mse_opinion==", "{:.5f}".format(outs[3]))
                                        b_mse_one.append(outs[1])
                                        u_mse_one.append(outs[2])
                                        bf_mse_one.append(outs[3])
                                        belief_one.append(outs[4])
                                        uncertain_one.append(outs[5])

                                print("Optimization Finished!")
                                print("run time = ", time.time()-t1)

                            with open(fileName,'rb') as pkl_file:
                                [V, E, Obs, E_X, X_b] = pickle.load(pkl_file)
                            sign_grad={}
                            for i in range(FLAGS.epochs):
                                for e_id,grads in enumerate(gradients[i][0][0]):
                                    edge_id=E[e_id]
                                    if test_mask[e_id]==False:
                                        if np.sum(np.sign(grads))>=0.0:
                                            # print(">=0",grads)
                                            if sign_grad.has_key(edge_id):
                                                sign_grad[edge_id].append(1.0)
                                            else:
                                                sign_grad[edge_id]=[1.0]
                                        else:
                                            # print("<0--------------------------",grads)
                                            if sign_grad.has_key(edge_id):
                                                sign_grad[edge_id].append(-1.0)
                                            else:
                                                sign_grad[edge_id]=[-1.0]
                            print(len(sign_grad))
                            

                            for gamma in [0.0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.20, 0.25][:]:  # 8

                                fout = out_folder + '/network_{}_weekday_{}_hour_{}_refspeed_{}-testratio-{}-gamma-{}-realization-{}.pkl'.format(
                                dataset, weekday, hour, ref_per, test_ratio, gamma, real_i)
                                print(fout)
                                if gamma>0.0:
                                    Obs_g=copy.deepcopy(Obs)
                                    for e in sign_grad.keys():
                                        for t in range(0, T):
                                            for i in range(len(sign_grad[e])):
                                                Obs_g[e][t] = clip01(Obs_g[e][t] + 0.01 * sign_grad[e][i])  # clip between [0,1]
                                            if np.abs(Obs_g[e][t] - Obs[e][t]) > gamma:
                                                Obs_g[e][t] = clip01(Obs[e][t] + np.sign(Obs_g[e][t] - Obs[e][t]) * gamma)  # clip |py_adv-py_orig|<gamma

                                    pkl_file = open(fout, 'wb')
                                    pickle.dump([V,E, Obs_g, E_X, X_b], pkl_file)
                                    pkl_file.close()
                                else:
                                    pkl_file = open(fout, 'wb')
                                    pickle.dump([V, E, Obs, E_X, X_b], pkl_file)
                                    pkl_file.close()


                            # print("belief:", np.mean(b_mse), "uncertain:", np.mean(u_mse), "time window = ", FLAGS.T, "test_rio = ", FLAGS.test_rat)
                            # sess.close()

