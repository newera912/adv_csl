__author__ = 'Feng Chen'
import pickle
from math import *
import numpy as np
import copy
import random
import networkx as nx
import numpy as np
# import matplotlib.pyplot as plt
from scipy.stats import beta
from cubic import cubic
from basic_funs import *
from network_funs import *
import os,time
import multiprocessing
from random import shuffle
from DataWrapper import *
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

def sample_X(test_ratio, V, E):
    n = len(V)
    n_E = len(E)
    rands = np.random.permutation(n_E)[:int(np.round(test_ratio * n_E))]
    edge_up_nns, edge_down_nns = get_edge_nns(V, E)
    E_X = {E[i]:1 for i in rands if edge_up_nns.has_key(E[i]) or edge_down_nns.has_key(E[i])}
    return E_X

def sample_X2(test_ratio, V, E):
    n = len(V)
    n_E = len(E)
    rands = np.random.permutation(n_E)[:]
    edge_up_nns, edge_down_nns = get_edge_nns(V, E)
    E_X={}
    count=0.0
    for i in rands:
        if edge_up_nns.has_key(E[i]) or edge_down_nns.has_key(E[i]):
            E_X[E[i]]=1
            count+=1.0
        if count>int(np.round(test_ratio * n_E)):
            break
    print test_ratio,count/len(E)
    return E_X

class Consumer(multiprocessing.Process):
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means we should exit
                # print '%s: Exiting' % proc_name
                break
            # print '%s: %s' % (proc_name, next_task)
            answer = next_task()
            self.result_queue.put(answer)
        return
class Task_generate(object):
    def __init__(self,data_root, dataset, hour, weekday, ref_per,test_ratio,ratio_conflict,real_i):
        self.data_root = data_root
        self.dataset = dataset
        self.hour = hour
        self.weekday = weekday
        self.ref_per = ref_per
        self.test_ratio = test_ratio
        self.ratio_conflict =ratio_conflict
        self.real_i = real_i

    def __call__(self):
        #s_data_root = "/network/rit/lab/ceashpc/adil/data/csl-data/Traffic/"+self.dataset
        # data_root = "/network/rit/lab/ceashpc/adil/data/csl-data/traffic"
        f = self.data_root + '/network_{}_weekday_{}_hour_{}_refspeed_{}-testratio-{}-confictratio-{}-realization-{}.pkl'.format(
            self.dataset, self.weekday, self.hour, self.ref_per, self.test_ratio, self.ratio_conflict,self.real_i)
        print f
        dw = DataWrapper(dataset=self.dataset, directed=True)
        V, E, Obs, _ = dw.get_data_case(self.hour, self.weekday, self.ref_per)
        # freq={}
        # for e,ob in Obs.items():
        #     pos=sum(ob)
        #     if freq.has_key(pos):
        #         freq[pos]+=1.0
        #     else:
        #         freq[pos]=1.0
        # for k in sorted(freq.keys()):
        #     print k,freq[k]
        #
        # time.sleep(10000)
        E_X = sample_X2(self.test_ratio, V, E)

        E_Y = [e for e in E if not E_X.has_key(e)]
        rand_seq_E_Y = copy.deepcopy(E_Y)
        shuffle(rand_seq_E_Y)
        cnt = int(np.round(len(E_Y) * self.ratio_conflict))
        X_b = rand_seq_E_Y[:cnt]
        T = len(Obs[E[0]])
        mid = int(np.round(T / 2.0))
        for e in X_b:
            for t in range(mid, T):
                Obs[e][t] = 1 - Obs[e][t]

        pkl_file = open(f, 'wb')
        pickle.dump([V, E, Obs, E_X, X_b], pkl_file)
        pkl_file.close()
        # print f

    def __str__(self):
        return '%s' % (self.p0)

"""
Generate simulation datasets with different graph sizes and rates
"""
def simulation_data_generator3():
    data_root="/network/rit/lab/ceashpc/adil/data/csl-data/Dec10/"


    tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()
    num_consumers = 50  # We only use 5 cores.
    # print 'Creating %d consumers' % num_consumers
    consumers = [Consumer(tasks, results)
                 for i in range(num_consumers)]
    for w in consumers:
        w.start()
    num_jobs=0
    realizations=10
    ref_pers = [0.6,0.7, 0.8]
    datasets = ['philly', 'dc']
    for ref_per in ref_pers[:]:
        for dataset in datasets:
            dataroot=data_root+dataset+"/"
            if not os.path.exists(dataroot):
                os.makedirs(dataroot)
            for weekday in range(5)[:]:
                for hour in range(8, 22):
                    for test_ratio in [0.1, 0.2, 0.3, 0.4, 0.5][1:2]:
                        for ratio_conflict in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6][3:4]:
                            for real_i in range(realizations)[:]:
                                tasks.put(Task_generate(dataroot,dataset, hour, weekday, ref_per,test_ratio,ratio_conflict,real_i))
                                num_jobs += 1
    for i in range(num_consumers):
        tasks.put(None)

    while num_jobs:
        results.get()
        num_jobs -= 1
        print num_jobs





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
    # initial positive users
    for i in np.random.permutation(n_E)[:m]:
        obs[E[i]] = 1
    print "# edges: {} # positives: {}".format(n_E, np.sum(obs.values()))
    # draw_graph(V, E, {k: [val] for k, val in obs.items()})
    feat_data = {e: [] for e in E}
    for feat_i in range(n_features):
        if swap_ratio!=0.0:
            obs = pertubation(obs, swap_ratio)
        obs = stablization(obs, nns)
        for (e, feat) in obs.items():
            feat_data[e].append(feat)
        print "# edges: {} # positives: {}".format(n_E, np.sum([item[0] for item in feat_data.values()]))
    return feat_data




def main():
    # filename = "data/trust-analysis/nodes-47676.pkl"
    # pkl_file = open(filename, 'rb')
    # [V, E] = pickle.load(pkl_file)
    # print len(V), len(E)
    # simulation_data_generator2()
    simulation_data_generator3()

if __name__=='__main__':
    main()

