import numpy as np
import random
from scipy import sparse
from collections import Counter
import read_bigdata as rb_data

def get_neighbor(adj):
    neigh = []
    for item in adj:
        neigh_i = []
        for i in range(np.size(item)):
            if item[i] == 1.0:
                neigh_i.append(i)
        neigh.append(neigh_i)
    return neigh


def generate_synthetic_belief(num, nosie):
    random.seed(123)
    adjacency = np.load("/network/rit/lab/ceashpc/xujiang/MILCOM/GCN_traffic/gcn/data/adjacency_matrix_dc_milcom.npy")
    adjacency_l = get_neighbor(adjacency)
    syn_feature = np.ones(len(adjacency_l)) * 0.1
    random_point = int(random.randrange(len(adjacency_l)))
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
            syn_feature[int(random_next[0])] = 0.8
        random_point = int(random_next[0])
    nosie_feat = np.array(syn_feature)
    noise_index = random.sample(range(len(adjacency_l)), int(nosie * len(adjacency_l)))
    for item in noise_index:
        item = int(item)
        nosie_feat[item] = 0.4
    return syn_feature, nosie_feat


def generate_synthetic_uncertain(num, nosie):
    random.seed(123)
    adjacency = np.load("/network/rit/lab/ceashpc/xujiang/MILCOM/GCN_traffic/gcn/data/adjacency_matrix_dc_milcom.npy")
    adjacency_l = get_neighbor(adjacency)
    syn_feature = np.ones(len(adjacency_l)) * 0.1
    random_point = int(random.randrange(len(adjacency_l)))
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
            syn_feature[int(random_next[0])] = 0.2
        random_point = int(random_next[0])
    nosie_feat = np.array(syn_feature)
    noise_index = random.sample(range(len(adjacency_l)), int(nosie * len(adjacency_l)))
    for item in noise_index:
        item = int(item)
        nosie_feat[item] = 0.4
    return syn_feature, nosie_feat


def generate_synthetic_belief2(num, nosie):
    random.seed(123)
    adjacency = np.load("/network/rit/lab/ceashpc/xujiang/MILCOM/GCN_traffic/gcn/data/adjacency_matrix_ph_milcom.npy")
    adjacency_l = get_neighbor(adjacency)
    syn_feature = np.ones(len(adjacency_l)) * 0.1
    random_point = int(random.randrange(len(adjacency_l)))
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
            syn_feature[int(random_next[0])] = 0.8
        random_point = int(random_next[0])
    nosie_feat = np.copy(syn_feature)
    noise_index = random.sample(range(len(adjacency_l)), int(nosie * len(adjacency_l)))
    for item in noise_index:
        item = int(item)
        if item in random_data:
            nosie_feat[item] = 0.1
        else:
            nosie_feat[item] = 0.8

    return syn_feature, nosie_feat


def generate_synthetic_uncertain2(num, nosie):
    random.seed(123)
    adjacency = np.load("/network/rit/lab/ceashpc/xujiang/MILCOM/GCN_traffic/gcn/data/adjacency_matrix_ph_milcom.npy")
    adjacency_l = get_neighbor(adjacency)
    syn_feature = np.ones(len(adjacency_l)) * 0.1
    random_point = int(random.randrange(len(adjacency_l)))
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
            syn_feature[int(random_next[0])] = 0.2
        random_point = int(random_next[0])
    nosie_feat = np.copy(syn_feature)
    noise_index = random.sample(range(len(adjacency_l)), int(nosie * len(adjacency_l)))
    for item in noise_index:
        item = int(item)
        if item in random_data:
            nosie_feat[item] = 0.1
        else:
            nosie_feat[item] = 0.2
    return syn_feature, nosie_feat


def get_omega(b, u):
    W = 2.0
    a = 0.5
    d = 1.0 - b - u
    r = W * b / u
    s = W * d / u
    alpha = r + W * a
    beta = s + W * (1.0 - a)
    omega = alpha / (alpha + beta)
    return omega


def knn():
    test_num = 600
    adjacency = np.load("/network/rit/lab/ceashpc/xujiang/MILCOM/GCN_traffic/gcn/data/adjacency_matrix_dc_milcom.npy")
    adjacency_l = get_neighbor(adjacency)
    random.seed(132)
    test_index = random.sample(range(len(adjacency)), test_num)
    _, feat_b = generate_synthetic_belief2(500, 0.1)
    _, feat_u = generate_synthetic_uncertain2(500, 0.1)
    pred_b = np.zeros_like(feat_b)
    pred_u = np.zeros_like(feat_u)
    b_truth = np.copy(feat_b)
    u_truth = np.copy(feat_u)
    test_mask = np.zeros_like(feat_b)
    for i in test_index:
        test_mask[i] = 1.0
        feat_b[i] = 0.0
        feat_u[i] = 0.0
    mean_b = np.sum(feat_b) / (len(feat_b) - test_num)
    mean_u = np.sum(feat_u) / (len(feat_u) - test_num)
    for i in test_index:
        feat_b[i] = mean_b
        feat_u[i] = mean_u
    for i in range(len(feat_b)):
        neigh = adjacency[i]
        pred_b[i] = np.mean(feat_b * neigh) / np.mean(neigh)
        pred_u[i] = np.mean(feat_u * neigh) / np.mean(neigh)
    test_mask /= np.mean(test_mask)
    b_mse = np.mean(np.abs(pred_b - b_truth) * test_mask)
    u_mse = np.mean(np.abs(pred_u - u_truth) * test_mask)
    omega_t = get_omega(b_truth, u_truth)
    omega_p = get_omega(pred_b, pred_u)
    eb_mse = np.mean(np.abs(omega_p - omega_t) * test_mask)
    return b_mse, u_mse, eb_mse


def knn2():
    test_rat = 0.9
    adjacency = np.load("/network/rit/lab/ceashpc/xujiang/MILCOM/GCN_traffic/gcn/data/adjacency_matrix_dc_milcom.npy")
    adjacency_l = get_neighbor(adjacency)
    test_num = int(test_rat * len(adjacency))
    random.seed(132)
    test_index = random.sample(range(len(adjacency)), test_num)
    feat_b, feat_u = rb_data.get_dc_data()
    pred_b = np.zeros_like(feat_b)
    pred_u = np.zeros_like(feat_u)
    b_truth = np.copy(feat_b)
    u_truth = np.copy(feat_u)
    test_mask = np.zeros_like(feat_b)
    for i in test_index:
        test_mask[i] = 1.0
        feat_b[i] = 0.0
        feat_u[i] = 0.0
    mean_b = np.sum(feat_b) / (len(feat_b) - test_num)
    mean_u = np.sum(feat_u) / (len(feat_u) - test_num)
    # for i in test_index:
    #     feat_b[i] = mean_b
    #     feat_u[i] = mean_u
    for i in range(len(feat_b)):
        neigh1 = adjacency_l[i]
        neigh = adjacency[i]
        k = 0.0
        for n in neigh1:
            if n in test_index:
                pass
            else:
                k += 1.0
        if k > 0.0:
            pred_b[i] = np.sum(feat_b * neigh) / k
            pred_u[i] = np.sum(feat_u * neigh) / k
        else:
            pred_b[i] = mean_b
            pred_u[i] = mean_u
    test_mask /= np.mean(test_mask)
    b_mse = np.mean(np.abs(pred_b - b_truth) * test_mask)
    u_mse = np.mean(np.abs(pred_u - u_truth) * test_mask)
    omega_t = get_omega(b_truth, u_truth)
    omega_p = get_omega(pred_b, pred_u)
    eb_mse = np.mean(np.abs(omega_p - omega_t) * test_mask)
    return b_mse, u_mse, eb_mse


def knn_pa(index):
    test_rat = 0.8
    adjacency = np.load("/network/rit/lab/ceashpc/xujiang/MILCOM/GCN_traffic/gcn/data/adjacency_matrix_ph_milcom.npy")
    adjacency_l = get_neighbor(adjacency)
    test_num = int(test_rat * len(adjacency))
    random.seed(132)
    test_index = random.sample(range(len(adjacency)), test_num)
    b_all = np.load("/network/rit/lab/ceashpc/xujiang/project/GAE_TEST/gae/traffic_data/pa_belief_0.9.npy")
    u_all = np.load("/network/rit/lab/ceashpc/xujiang/project/GAE_TEST/gae/traffic_data/pa_uncertain_0.9.npy")
    feat_b = b_all[index]
    feat_u = u_all[index]
    pred_b = np.zeros_like(feat_b)
    pred_u = np.zeros_like(feat_u)
    b_truth = np.copy(feat_b)
    u_truth = np.copy(feat_u)
    test_mask = np.zeros_like(feat_b)
    for i in test_index:
        test_mask[i] = 1.0
        feat_b[i] = 0.0
        feat_u[i] = 0.0
    mean_b = np.sum(feat_b) / (len(feat_b) - test_num)
    mean_u = np.sum(feat_u) / (len(feat_u) - test_num)
    # for i in test_index:
    #     feat_b[i] = mean_b
    #     feat_u[i] = mean_u
    for i in range(len(feat_b)):
        neigh1 = adjacency_l[i]
        neigh = adjacency[i]
        k = 0.0
        for n in neigh1:
            if n in test_index:
                pass
            else:
                k += 1.0
        if k > 0.0:
            pred_b[i] = np.sum(feat_b * neigh) / k
            pred_u[i] = np.sum(feat_u * neigh) / k
        else:
            pred_b[i] = mean_b
            pred_u[i] = mean_u
    test_mask /= np.mean(test_mask)
    b_mse = np.mean(np.abs(pred_b - b_truth) * test_mask)
    u_mse = np.mean(np.abs(pred_u - u_truth) * test_mask)
    omega_t = get_omega(b_truth, u_truth)
    omega_p = get_omega(pred_b, pred_u)
    eb_mse = np.mean(np.abs(omega_p - omega_t) * test_mask)
    return b_mse, u_mse, eb_mse


def knn_dc(index):
    test_rat = 0.8
    adjacency = np.load("/network/rit/lab/ceashpc/xujiang/MILCOM/GCN_traffic/gcn/data/adjacency_matrix_dc_milcom.npy")
    adjacency_l = get_neighbor(adjacency)
    test_num = int(test_rat * len(adjacency))
    random.seed(132)
    test_index = random.sample(range(len(adjacency)), test_num)
    b_all = np.load("/network/rit/lab/ceashpc/xujiang/project/GAE_TEST/gae/traffic_data/dc_belief_0.9.npy")
    u_all = np.load("/network/rit/lab/ceashpc/xujiang/project/GAE_TEST/gae/traffic_data/dc_uncertain_0.9.npy")
    feat_b = b_all[index]
    feat_u = u_all[index]
    pred_b = np.zeros_like(feat_b)
    pred_u = np.zeros_like(feat_u)
    b_truth = np.copy(feat_b)
    u_truth = np.copy(feat_u)
    test_mask = np.zeros_like(feat_b)
    for i in test_index:
        test_mask[i] = 1.0
        feat_b[i] = 0.0
        feat_u[i] = 0.0
    mean_b = np.sum(feat_b) / (len(feat_b) - test_num)
    mean_u = np.sum(feat_u) / (len(feat_u) - test_num)
    # for i in test_index:
    #     feat_b[i] = mean_b
    #     feat_u[i] = mean_u
    for i in range(len(feat_b)):
        neigh1 = adjacency_l[i]
        neigh = adjacency[i]
        k = 0.0
        for n in neigh1:
            if n in test_index:
                pass
            else:
                k += 1.0
        if k > 0.0:
            pred_b[i] = np.sum(feat_b * neigh) / k
            pred_u[i] = np.sum(feat_u * neigh) / k
        else:
            pred_b[i] = mean_b
            pred_u[i] = mean_u
    test_mask /= np.mean(test_mask)
    b_mse = np.mean(np.abs(pred_b - b_truth) * test_mask)
    u_mse = np.mean(np.abs(pred_u - u_truth) * test_mask)
    omega_t = get_omega(b_truth, u_truth)
    omega_p = get_omega(pred_b, pred_u)
    eb_mse = np.mean(np.abs(omega_p - omega_t) * test_mask)
    return b_mse, u_mse, eb_mse

def knn_beijing():

    # adjacency = np.load("/network/rit/lab/ceashpc/xujiang/project/GAE_TEST/gae/traffic_data/adj_undirect_beijing.npy")
    adjacency_l = np.load("20130915ref_proc-0.9_neigh.npy")
    test_num = int(0.4 * len(adjacency_l))
    random.seed(132)
    test_index = random.sample(range(len(adjacency_l)), test_num)
    feat_b = np.load("belief_undirect_beijing.npy")
    feat_u = np.load("uncertain_undirect_beijing.npy")
    pred_b = np.zeros_like(feat_b)
    pred_u = np.zeros_like(feat_u)
    b_truth = np.copy(feat_b)
    u_truth = np.copy(feat_u)
    test_mask = np.zeros_like(feat_b)
    for i in test_index:
        test_mask[i] = 1.0
        feat_b[i] = 0.0
        feat_u[i] = 0.0
    mean_b = np.sum(feat_b) / (len(feat_b) - test_num)
    mean_u = np.sum(feat_u) / (len(feat_u) - test_num)
    # for i in test_index:
    #     feat_b[i] = mean_b
    #     feat_u[i] = mean_u
    for i in range(len(feat_b)):
        neigh1 = adjacency_l[i]
        neigh1 = np.setdiff1d(neigh1, [i])
        # neigh = adjacency[i]
        k = 0.0
        for n in neigh1:
            if n in test_index:
                pass
            else:
                k += 1.0
                pred_b[i] += feat_b[n]
                pred_u[i] += feat_u[n]
        if k > 0.0:
            pred_b[i] = pred_b[i] / k
            pred_u[i] = pred_u[i] / k
        else:
            pred_b[i] = mean_b
            pred_u[i] = mean_u
    test_mask /= np.mean(test_mask)
    b_mse = np.mean(np.abs(pred_b - b_truth) * test_mask)
    u_mse = np.mean(np.abs(pred_u - u_truth) * test_mask)
    omega_t = get_omega(b_truth, u_truth)
    omega_p = get_omega(pred_b, pred_u)
    eb_mse = np.mean(np.abs(omega_p - omega_t) * test_mask)
    return b_mse, u_mse, eb_mse


if __name__ == '__main__':
    belief = []
    uncertain = []
    opinion_error = []
    for k in range(30):
        b, u, o = knn_dc(k)
        print(b, u, o)
        belief.append(b)
        uncertain.append(u)
        opinion_error.append(o)
    print("belief:", np.mean(belief), "uncertain:", np.mean(uncertain), "opinion:", np.mean(opinion_error))
