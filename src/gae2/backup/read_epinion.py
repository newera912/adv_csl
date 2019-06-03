import numpy as np
import pickle
from operator import add
import random


def read_eopinion_data():
    methods = ["csl", "sl", "base1", "base2", "base3"][:]
    graph_sizes = [500, 1000, 5000, 10000, 47676]
    # graph_sizes = [2500, 7500]
    ratios = [0.05, 0.10, 0.2, 0.3, 0.4]
    realizations = 1
    feat_eopinion = []
    test_index = []
    for swap_ratio in [0.01, 0.05][1:2]:
        for test_ratio in [0.1, 0.2, 0.3, 0.4][0:1]:
            for T in [2, 3, 6, 8, 11][3:4]:
                for ratio in ratios[2:3]:  # 2 * 4 * 5 * 4 = 160zzz
                    for graph_size in graph_sizes[3:4]:
                        for real_i in range(realizations)[:]:
                            for method in methods[:1]:
                                f = "/network/rit/lab/ceashpc/xujiang/eopinion/data/nodes-{}-rate-{}-testratio-{}-swaprate-{}-realization-{}-data-X.pkl".format(
                                    graph_size, ratio, test_ratio, swap_ratio, real_i)
                                pkl_file = open(f, 'rb')
                                [V, E, Obs, E_X] = pickle.load(pkl_file)
                                feat_hour = []
                                edge_double = []
                                for edge in E:
                                    edge_ = (edge[1], edge[0])
                                    if edge_ in E:
                                        E.remove(edge_)
                                        edge_double.append(edge)
                                for edge_d in edge_double:
                                    edge_d_ = (edge_d[1], edge_d[0])
                                    Obs[edge_d] = map(add, Obs[edge_d], Obs[edge_d_])
                                for i in range(len(E)):
                                    edge = E[i]
                                    edge_ = (edge[1], edge[0])
                                    feat_hour.append(Obs[edge])
                                    if edge in E_X:
                                        test_index.append(i)
                                    elif edge_ in E_X:
                                        test_index.append(i)
                                ad_m = np.zeros([len(E), len(E)])
                                neigh = []
                                for i in range(len(E)):
                                    neigh_i = find_neigh_edge(E, i)
                                    neigh.append(neigh_i)
                                    for k in neigh_i:
                                        k = int(k)
                                        ad_m[i][k] = int(1)
                                feat_eopinion.append(feat_hour)
                                # np.save("/network/rit/lab/ceashpc/xujiang/eopinion/neigh_10000.npy", neigh)
                                # np.save("/network/rit/lab/ceashpc/xujiang/eopinion/adjacency_matrix_10000.npy", ad_m)
    return feat_eopinion, test_index


def find_neigh_edge(E, i):
    neigh = []
    nodes = E[i]
    for j in range(len(E)):
        if j != i:
            for node in nodes:
                if node in E[j]:
                    neigh.append(j)
                    break
    return neigh


def get_ad_matrix():
    methods = ["csl", "sl", "base1", "base2", "base3"][:]
    graph_sizes = [500, 1000, 5000, 10000, 47676, 2500, 7500]
    # graph_sizes = [2500, 7500]
    ratios = [0.05, 0.10, 0.2, 0.3, 0.4]
    realizations = 1
    feat_eopinion = []
    f = ""
    for swap_ratio in [0.01, 0.05][1:2]:
        for test_ratio in [0.1, 0.2, 0.3, 0.4][1:2]:
            for T in [2, 3, 6, 8, 11][2:3]:
                for ratio in ratios[2:3]:  # 2 * 4 * 5 * 4 = 160zzz
                    for graph_size in graph_sizes[0:1]:
                        for real_i in range(realizations)[:]:
                            for method in methods[:1]:
                                f = "/network/rit/lab/ceashpc/fenglab/baojian/git/subjectiveLogic/eopinion/data/trust-analysis2/nodes-{}-rate-{}-testratio-{}-swaprate-{}-realization-{}-data-X.pkl".format(
                                    graph_size, ratio, test_ratio, swap_ratio, real_i)
    pkl_file = open(f, 'rb')
    # pkl_file = open("/network/rit/lab/ceashpc/fenglab/baojian/git/subjectiveLogic/eopinion/data/trust-analysis2/nodes-7500-T-6-rate-0.2-testratio-0.2-swaprate-0.05-realization-0-data-X.pkl")
    [V, E, Obs, E_X] = pickle.load(pkl_file)
    ad_m = np.zeros([len(E), len(E)])
    for i in range(len(E)):
        neigh_i = find_neigh_edge(E, i)
        for k in neigh_i:
            k = int(k)
            ad_m[i][k] = int(1)

    return ad_m


def test_():
    methods = ["csl", "sl", "base1", "base2", "base3"][:]
    graph_sizes = [500, 1000, 5000, 10000, 47676, 2500, 7500]
    # graph_sizes = [2500, 7500]
    ratios = [0.05, 0.10, 0.2, 0.3, 0.4]
    realizations = 1
    feat_eopinion = []
    f = ""
    for swap_ratio in [0.01, 0.05][1:2]:
        for test_ratio in [0.1, 0.2, 0.3, 0.4][1:2]:
            for T in [2, 3, 6, 8, 11][2:3]:
                for ratio in ratios[2:3]:  # 2 * 4 * 5 * 4 = 160zzz
                    for graph_size in graph_sizes[0:1]:
                        for real_i in range(realizations)[:]:
                            for method in methods[:1]:
                                f = "/network/rit/lab/ceashpc/fenglab/baojian/git/subjectiveLogic/eopinion/data/trust-analysis2/nodes-{}-rate-{}-testratio-{}-swaprate-{}-realization-{}-data-X.pkl".format(
                                    graph_size, ratio, test_ratio, swap_ratio, real_i)
    pkl_file = open(f, 'rb')
    # pkl_file = open("/network/rit/lab/ceashpc/fenglab/baojian/git/subjectiveLogic/eopinion/data/trust-analysis2/nodes-7500-T-6-rate-0.2-testratio-0.2-swaprate-0.05-realization-0-data-X.pkl")
    [V, E, Obs, E_X] = pickle.load(pkl_file)
    for edge in E:
        edge_ = (edge[1], edge[0])
        if edge_ in E:
            E.remove(edge_)
    return


def get_E_X():
    random.seed(132)
    feat_ = np.load("/network/rit/lab/ceashpc/xujiang/eopinion/graph_500_test_0.2.npz")
    feature_hour = feat_["arr_0"]
    # feature_hour = features_[hour]
    feature_edge_i = feature_hour[0][:, 1]
    # test_index = random.sample(range(len(feature_edge_i)), 100)

    methods = ["csl", "sl", "base1", "base2", "base3"][:]
    graph_sizes = [500, 1000, 5000, 10000, 47676]
    ratios = [0.05, 0.10, 0.2, 0.3, 0.4]
    realizations = 1

    for swap_ratio in [0.01, 0.05][1:2]:
        for test_ratio in [0.1, 0.2, 0.3, 0.4][1:2]:
            for T in [2, 3, 6, 8, 11][3:4]:
                for ratio in ratios[2:3]:  # 2 * 4 * 5 * 4 = 160zzz
                    for graph_size in graph_sizes[3:4]:
                        for real_i in range(realizations)[:]:
                            for method in methods[:1]:
                                f = "/network/rit/lab/ceashpc/fenglab/baojian/git/subjectiveLogic/eopinion/data/trust-analysis2/nodes-{}-rate-{}-testratio-{}-swaprate-{}-realization-{}-data-X.pkl".format(
                                    graph_size, ratio, test_ratio, swap_ratio, real_i)
                                pkl_file = open(f, 'rb')
                                [V, E, Obs, E_X] = pickle.load(pkl_file)

                                E_X = []
                                E1 = [x for x in E]
                                for edge in E1:
                                    edge_ = (edge[1], edge[0])
                                    if edge_ in E1:
                                        E1.remove(edge_)
                                test_index = random.sample(range(len(E1)), 4000)
                                for i in test_index:
                                    E_X.append(E1[i])
                                pkl_file.close()
                                f2 = "/network/rit/lab/ceashpc/xujiang/eopinion/data/nodes-{}-rate-{}-testratio-{}-swaprate-{}-realization-{}-data-X.pkl".format(
                                    graph_size, ratio, 0.4, swap_ratio, real_i)
                                pkl_file = open(f2, 'wb')
                                pickle.dump([V, E, Obs, E_X], pkl_file)
                                pkl_file.close()
                                print(1)
    return


if __name__ == '__main__':
    # aa = get_ad_matrix()
    # np.save("/network/rit/home/xz381633/traffic_deep/input_data/adjacency_matrix_eopinion_7500.npy", aa)
    # ff, tt = read_eopinion_data()
    # np.savez("/network/rit/lab/ceashpc/xujiang/eopinion/graph_10000_test_0.2.npz", ff, tt)
    # feat_ = np.load("/network/rit/lab/ceashpc/xujiang/eopinion/graph_10000_test_0.2.npz")
    # feat = feat_["arr_0"]
    # sample_index = feat_["arr_1"]
    # print(sample_index)
    read_eopinion_data()
    print(1)
