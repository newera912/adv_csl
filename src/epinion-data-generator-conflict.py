__author__ = 'Feng Chen'
import pickle
from math import *
import numpy as np
import copy
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from cubic import cubic
import os
"""
INPUT
a: base with default 0.5
W:

OUTPUT
a randomly generated opinion (alpha, beta)
"""
def generate_a_random_opinion(a = 0.5, W = 1):
    W = 1
    u = random.uniform(0.01, 0.5)
    b = random.uniform(0.01, 1 - u)
    d = 1 - u - b
    r = b * W / u
    s = d * W / u
    alpha = r + a * W
    beta = s + (1-a) * W
    return (alpha, beta)

"""
INPUT
V = [0, 1, ...] is a list of vertex ids
E: a list of pairs of vertex ids
Obs: a dictionary with key edge and its value a list of congestion observations of this edge from t = 1 to t = T
Omega: a dictionary with key edge and its value a pair of alpha and beta
E_X: a list of pairs of vertex ids whose opinions will be predicted.

OUTPUT
omega_X: {edge: (alpha, beta), ...}, where edge \in E_X and is represented by a pair of vertex ids
"""
def SL_prediction(V, E, Omega, E_X):
    omega_X = {e: alpha_beta for e, alpha_beta in Omega.items() if e in E_X}
    return omega_X


"""
INPUT
V: a list of vertex ids
E: a list of ordered pairs of vertex ids
T: The number of observations. 100 by default.

OUTPUT
datasets: key is rate and the value is [[V, E, Omega, Obs, X], ...], where rate refers to the rate of edges that are randomly selected as target edges

"""
def simulation_data_generator(V, E, rates = [0.05, 0.1, 0.15, 0.25, 0.3, 0.4, 0.5], T = 50):
    datasets = {}
    len_E = len(E)
    Omega = {}
    realizations = 10
    for e in E:
        Omega[e] = generate_a_random_opinion()
    for rate in rates:
        rate_datasets = []
        for real_i in range(realizations):
            len_X = int(round(len_E * rate))
            rate_X = [E[i] for i in np.random.permutation(len_X)[:len_X]]
            rate_omega_X = SL_prediction(V, E, Omega, rate_X)
            rate_omega = copy.deepcopy(Omega)
            for e, alpha_beta in rate_omega_X.items():
                Omega[e] = alpha_beta
            rate_Obs = {}
            for e in E:
                e_alpha, e_beta = Omega[e]
                rate_Obs[e] = beta.rvs(e_alpha, e_beta, 0, 1, T)
            datasets[(rate, real_i)] = [V, E, rate_omega, rate_Obs, rate_X]
    return datasets




"""
Generate simulation datasets with different graph sizes and rates
"""
def simulation_data_generator2():
    realizations = 10
    graph_sizes = [500, 1000, 5000, 10000, 47676]
    # graph_sizes = [2500, 7500]
    ratios = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    for T in [2, 3, 6, 8, 11,20][5:]:
        for swap_ratio in [0.00, 0.01, 0.05][:1]:
            for ratio in ratios[:]:  #the percentage of edges set the observations to 1
                for graph_size in graph_sizes[:]:
                    for real_i in range(realizations)[:1]:
                        filename = "/home/apdm02/workspace/git/data/cls_conflict/trust-analysis/nodes-{0}.pkl".format(graph_size)
                        fout= "/home/apdm02/workspace/git/data/cls_conflict/trust-analysis3/nodes-{}-T-{}-rate-{}-swaprate-{}-realization-{}-data.pkl".format(graph_size, T, ratio, swap_ratio, real_i)
                        if not os.path.exists(fout):
                           print "--------- reading {}".format(filename)
                           pkl_file = open("/home/apdm02/workspace/git/data/cls_conflict/trust-analysis/nodes-{}.pkl".format(graph_size), 'rb')
                           [V, E] = pickle.load(pkl_file)
                           pkl_file.close()
                           print "--------- generating simulation data"
                           Obs = graph_process(V, E, T, ratio, swap_ratio)
                           # draw_graph(V, E, Obs)
                           print "---------------------graph size: {0}, rate: {1}, realization: {2}".format(graph_size, ratio, real_i)
                           # fout= "data/trust-analysis2/nodes-{}-rate-{}-swaprate-{}-realization-{}-data.pkl".format(graph_size, ratio, swap_ratio, real_i)
                           pkl_file = open(fout, 'wb')
                           pickle.dump([V, E, Obs], pkl_file)
                           pkl_file.close()

def gen_nns(V, E):
    nns = {v: [] for v in V}
    for (v1,v2) in E:
        if nns.has_key(v1):
            nns[v1].append(v2)
    return nns

def get_paths(e, nns):
    # print "e", e
    (v1, v2) = e
    paths = []
    v1_nns = nns[v1]
    for v1_n in v1_nns:
        if v2 in nns[v1_n]:
            paths.append([(v1, v1_n), (v1_n, v2)])
    return paths


def stablization(obs, nns):
    iterations = 10000
    for iter in range(iterations):
        odds = []
        e = obs.keys()[int(np.floor(len(obs) * random.random()))]
        for path in get_paths(e, nns):
            if obs[path[0]] == 1 and obs[path[1]] == 1:
                odds.append(1)
            else:
                odds.append(0)
            # elif obs[path[0]] == 0 and obs[path[1]] == 0:
            #     odds.append(0)
        if len(odds) > 0:
            odd = random.choice(odds)
            if odd == 1 or random.random() < 0.00:
                obs[e] = 1
    return obs


"""
Randomly select swap_ratio edges and swap their observations (0 -> 1, 1 -> 1)
"""
def pertubation(obs, swap_ratio):
    for e, feat in obs.items():
        if random.random() < swap_ratio:
            obs[e] = 1 - feat
    return obs


"""
INPUT
V
E
ratio: percentage of links that are trusted in the beginning

OUTPUT
feat_data: a dictionary in which a key is a edge and the value is a list of observations.
"""
def graph_process(V, E, n_features, ratio, swap_ratio):
    nns = gen_nns(V, E)
    n_E = len(E)
    m = int(np.floor(n_E * ratio))
    obs = {e: 0 for e in E}
    for i in np.random.permutation(n_E)[:m]:
        obs[E[i]] = 1
    print "# edges: {} # positives: {}".format(n_E, np.sum(obs.values()))
    # draw_graph(V, E, {k: [val] for k, val in obs.items()})
    feat_data = {e: [] for e in E}
    for feat_i in range(n_features):
        obs = pertubation(obs, swap_ratio)
        obs = stablization(obs, nns)
        for (e, feat) in obs.items():
            feat_data[e].append(feat)
        print "# edges: {} # positives: {}".format(n_E, np.sum([item[0] for item in feat_data.values()]))
    return feat_data


"""
Generate simulation datasets with different graph sizes and rates
"""
def simulation_data_generator1():
    graph_sizes = [100, 500, 1000, 5000, 10000, 47676]
    rates = [0.05, 0.1, 0.15, 0.25, 0.3, 0.4, 0.5]
    T = 10
    for graph_size in graph_sizes[0:2]:
        filename = "data/trust-analysis/nodes-{0}.pkl".format(graph_size)
        print "--------- reading {0}".format(filename)
        pkl_file = open("data/trust-analysis/nodes-{0}.pkl".format(graph_size), 'rb')
        [V, E] = pickle.load(pkl_file)
        print len(V), len(E)
        pkl_file.close()
        print "--------- generating simulation data"
        datasets = simulation_data_generator(V, E, rates[:1], T)
        for (rate, real_i), dataset in datasets.items():
            print "---------------------graph size: {0}, rate: {1}, realization: {2}".format(graph_size, rate, real_i)
            pkl_file = open("data/trust-analysis/nodes-{0}-rate-{1}-realization-{2}-data.pkl".format(graph_size, rate, real_i), 'wb')
            pickle.dump(dataset, pkl_file)
            pkl_file.close()


def draw_graph(V, E, Obs):
    G = nx.Graph()
    G.add_nodes_from(V)
    for e in E:
        G.add_edge(e[0],e[1],weight = Obs[e][0])
    nx.draw(G)
    nx.draw(G,pos=nx.spectral_layout(G), nodecolor='r',edge_color='b')
    plt.show()
    #
    #
    # partition = community.best_partition(G)
    # size = float(len(set(partition.values())))
    # pos = nx.spring_layout(G)
    # count = 0.
    # for com in set(partition.values()) :
    #     count = count + 1.
    #     list_nodes = [nodes for nodes in partition.keys()
    #                                 if partition[nodes] == com]
    #     nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
    #                                 node_color = str(count / size))
    #
    #
    # nx.draw_networkx_edges(G,pos, alpha=0.5)


def main():
    # filename = "data/trust-analysis/nodes-47676.pkl"
    # pkl_file = open(filename, 'rb')
    # [V, E] = pickle.load(pkl_file)
    # print len(V), len(E)
    simulation_data_generator2()


if __name__=='__main__':
    main()

