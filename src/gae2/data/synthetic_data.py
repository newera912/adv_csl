import numpy as np
import random
from scipy import sparse
from collections import Counter


def get_adjacency_matrix():
    edge_neighbor = np.load("/network/rit/home/xz381633/traffic_deep/input_data/edge_neighbor.npz")
    u_neigh = edge_neighbor["arr_0"]
    v_neigh = edge_neighbor["arr_1"]
    adjacency_matrix = np.zeros([len(u_neigh), len(u_neigh)])
    for i in range(len(u_neigh)):
        neigh = np.hstack((u_neigh[i], v_neigh[i]))
        for j in neigh:
            j = int(j)
            adjacency_matrix[i][j] = int(1)
    return adjacency_matrix


def get_hotvalue_f(feat):
    hotvalue = np.zeros([len(feat), 2])
    feat_n = np.zeros([len(feat), 3])
    for i in range(len(feat)):
        if feat[i] == -1.0:  # non-conjestion
            feat_n[i] = [0, 0, 1]
            hotvalue[i] = [0, 1]
        elif feat[i] == 1.0:  # conjestion
            feat_n[i] = [1, 0, 0]
            hotvalue[i] = [1, 0]
        else:  # unknown node
            feat_n[i] = [0, 1, 0]
            hotvalue[i] = [0, 0]
    return hotvalue, feat_n


def get_hotvalue(feat):
    hotvalue = np.zeros([len(feat), 2])
    for i in range(len(feat)):
        if feat[i] == -1.0:  # non-conjestion
            # feat[i] = 100.0
            hotvalue[i] = [0, 1]
        elif feat[i] == 1.0:  # conjestion
            hotvalue[i] = [1, 0]
        else:  # unknown node
            # feat[i] = 50.0
            hotvalue[i] = [0, 0]
    return hotvalue, feat


def get_adjacency_list():
    edge_neighbor = np.load("/network/rit/home/xz381633/traffic_deep/input_data/edge_neighbor.npz")
    u_neigh = edge_neighbor["arr_0"]
    v_neigh = edge_neighbor["arr_1"]
    adjacency_list = []
    for i in range(len(u_neigh)):
        neigh = np.hstack((u_neigh[i], v_neigh[i]))
        adjacency_list.append(neigh)
    return adjacency_list


def generate_synthetic_data(num, nosie):
    random.seed(123)
    adjacency_l = get_adjacency_list()
    syn_feature = -np.ones(len(adjacency_l))
    random_point = int(random.randrange(1522))
    random_data = []
    k = 0
    while k < num:
        neigh = adjacency_l[random_point]
        random_next = random.sample(neigh, 1)
        if random_next[0] in random_data:
            pass
        else:
            random_data.append(random_next[0])
            k = k + 1
            syn_feature[int(random_next[0])] = 1.0
        random_point = int(random_next[0])
    nosie_feat = np.array(syn_feature)
    noise_index = random.sample(range(len(adjacency_l)), int(nosie * len(adjacency_l)))
    for item in noise_index:
        item = int(item)
        nosie_feat[item] = -syn_feature[item]
    return syn_feature, nosie_feat


def load_data_hour(week, test_num):

    adj_n = get_adjacency_matrix()
    _, feature_edge_i = generate_synthetic_data(500, 0.1)
    random.seed()
    test_index = random.sample(range(len(feature_edge_i)), test_num)
    label, feature_n = get_hotvalue_f(feature_edge_i)

    # feature_n = feature_edge_i
    y_train = np.zeros_like(label)
    y_test = np.zeros_like(label)
    train_mask = np.zeros_like(feature_edge_i, dtype=bool)
    test_mask = np.zeros_like(feature_edge_i, dtype=bool)
    for index in test_index:
        feature_n[index] = [0, 1, 0]
    for i in range(len(test_mask)):
        if i in test_index:
            y_test[i] = label[i]
            test_mask[i] = True
        else:
            y_train[i] = label[i]
            train_mask[i] = True
    y_val = y_test
    val_mask = test_mask
    adj = sparse.csr_matrix(adj_n)
    # feature_n = feature_random(feature_n, test_index)
    # features = sparse.csr_matrix(np.reshape(feature_n, [len(feature_n), 1]))
    features = sparse.csr_matrix(feature_n)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


if __name__ == '__main__':
    s_data, n_data = generate_synthetic_data(300, 0.0)
    fe = np.reshape(s_data, [len(s_data), 1])
    nf = np.reshape(n_data, [len(n_data), 1])
    print(Counter(s_data))
    print(Counter(n_data))
