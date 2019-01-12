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
from gen_traffic_adv_example import inference_apdm_format as inference_apdm_format_conflict_evidence
from gen_traffic_adv_example_csl import inference_apdm_format as inference_apdm_format_csl
from log import Log


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
class Task_generate_rf(object):
    def __init__(self,data_root, dataset, hour, weekday, ref_per,test_ratio,gamma,real_i):
        self.data_root = data_root
        self.dataset = dataset
        self.hour = hour
        self.weekday = weekday
        self.ref_per = ref_per
        self.test_ratio = test_ratio
        self.gamma =gamma
        self.real_i = real_i

    def __call__(self):
        #s_data_root = "/network/rit/lab/ceashpc/adil/data/csl-data/Traffic/"+self.dataset
        # data_root = "/network/rit/lab/ceashpc/adil/data/csl-data/traffic"
        f = self.data_root + '/network_{}_weekday_{}_hour_{}_refspeed_{}-testratio-{}-gamma-{}-realization-{}.pkl'.format(
            self.dataset, self.weekday, self.hour, self.ref_per, self.test_ratio, self.gamma,self.real_i)
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

        """Step 3: flip observations on the edges """
        E_Y = [e for e in E if not E_X.has_key(e)]
        rand_seq_E_Y = copy.deepcopy(E_Y)
        shuffle(rand_seq_E_Y)
        cnt = int(np.round(len(E_Y) * self.gamma))
        X_b = rand_seq_E_Y[:cnt]
        T = len(Obs[E[0]])
        # mid = int(np.round(T / 2.0))
        for e in X_b:
            for t in range(0, T):
                Obs[e][t] = 1 - Obs[e][t]

        pkl_file = open(f, 'wb')
        pickle.dump([V, E, Obs, E_X, X_b], pkl_file)
        pkl_file.close()
        # print f

    def __str__(self):
        return '%s' % (self.p0)

def clip01(x):
    if x<0.0: return 0.0
    elif x>1.0: return 1.0
    else: return x

class Task_generate_rn(object):
    def __init__(self,data_root, dataset, hour, weekday, ref_per,test_ratio,gamma,real_i):
        self.data_root = data_root
        self.dataset = dataset
        self.hour = hour
        self.weekday = weekday
        self.ref_per = ref_per
        self.test_ratio = test_ratio
        self.gamma =gamma
        self.real_i = real_i

    def __call__(self):
        #s_data_root = "/network/rit/lab/ceashpc/adil/data/csl-data/Traffic/"+self.dataset
        # data_root = "/network/rit/lab/ceashpc/adil/data/csl-data/traffic"
        f = self.data_root + '/network_{}_weekday_{}_hour_{}_refspeed_{}-testratio-{}-gamma-{}-realization-{}.pkl'.format(
            self.dataset, self.weekday, self.hour, self.ref_per, self.test_ratio, self.gamma,self.real_i)
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

        """Step 2: Add noise to observations on the edges """
        E_Y = [e for e in E if not E_X.has_key(e)]

        X_b = []
        """ |noise_vector_i| <= gamma"""
        noise_vector = np.random.uniform(low=-self.gamma, high=self.gamma, size=(len(E_Y),))

        T = len(Obs[E[0]])
        for i,e in enumerate(E_Y):
            for t in range(0, T):
                Obs[e][t] = clip01(Obs[e][t]+noise_vector[i])   #clip between [0,1]

        pkl_file = open(f, 'wb')
        pickle.dump([V, E, Obs, E_X, X_b], pkl_file)
        pkl_file.close()
        # print f

    def __str__(self):
        return '%s' % (self.p0)

class Task_generate_PGD(object):
    def __init__(self,data_root, dataset, hour, weekday, ref_per,test_ratio,gamma,real_i,alpha):
        self.data_root = data_root
        self.dataset = dataset
        self.hour = hour
        self.weekday = weekday
        self.ref_per = ref_per
        self.test_ratio = test_ratio
        self.gamma =gamma
        self.real_i = real_i
        self.alpha=alpha

    def __call__(self):
        #s_data_root = "/network/rit/lab/ceashpc/adil/data/csl-data/Traffic/"+self.dataset
        # data_root = "/network/rit/lab/ceashpc/adil/data/csl-data/traffic"
        f = self.data_root + '/network_{}_weekday_{}_hour_{}_refspeed_{}-testratio-{}-gamma-{}-realization-{}.pkl'.format(
            self.dataset, self.weekday, self.hour, self.ref_per, self.test_ratio, self.gamma,self.real_i)
        print "File:",f
        dw = DataWrapper(dataset=self.dataset, directed=True)
        V, E, Obs, _ = dw.get_data_case(self.hour, self.weekday, self.ref_per)
        ObsO=copy.deepcopy(Obs)
        T = len(Obs[E[0]])
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

        """Step 2: Add noise to observations on the edges """
        E_Y = [e for e in E if not E_X.has_key(e)]

        X_b = []
        """ |p_y+alpha*sign(nabla_py L)| <= gamma"""
        if self.gamma > 0.0:
            sign_grad_py = gen_adv_exmaple(V, E, Obs, X_b, E_X)


            for e in E_Y:
                # print type(sign_grad_py[0])
                if e not in sign_grad_py[0].keys(): print "Eroorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr!"
                for t in range(0, T):
                    for i in range(len(sign_grad_py[t][e])):
                        Obs[e][t] = clip01(Obs[e][t] + self.alpha * sign_grad_py[t][e][i])  # clip between [0,1]
                    if np.abs(Obs[e][t] - ObsO[e][t]) > self.gamma:
                        Obs[e][t] = clip01(ObsO[e][t] + np.sign(
                            Obs[e][t] - ObsO[e][t]) * self.gamma)  # clip |py_adv-py_orig|<gamma
            print "Iteration Number", [len(sign_grad_py[i][sign_grad_py[i].keys()[0]]) for i in
                                       range(len(sign_grad_py))]

        """   V, E, Obs, Omega, b, X_b, E_X, logging, psl   """

        pkl_file = open(f, 'wb')
        pickle.dump([V, E, Obs, E_X, X_b], pkl_file)
        pkl_file.close()
        # print f

    def __str__(self):
        return '%s' % (self.p0)

class Task_generate_PGD_csl(object):
    def __init__(self,data_root, dataset, hour, weekday, ref_per,test_ratio,gamma,real_i,alpha,org_fileName):
        self.data_root = data_root
        self.dataset = dataset
        self.hour = hour
        self.weekday = weekday
        self.ref_per = ref_per
        self.test_ratio = test_ratio
        self.gamma =gamma
        self.real_i = real_i
        self.alpha=alpha
        self.org_fileName=org_fileName

    def __call__(self):
        #s_data_root = "/network/rit/lab/ceashpc/adil/data/csl-data/Traffic/"+self.dataset
        # data_root = "/network/rit/lab/ceashpc/adil/data/csl-data/traffic"
        f = self.data_root + '/network_{}_weekday_{}_hour_{}_refspeed_{}-testratio-{}-gamma-{}-realization-{}.pkl'.format(
            self.dataset, self.weekday, self.hour, self.ref_per, self.test_ratio, self.gamma,self.real_i)
        print "File:",f
        # dw = DataWrapper(dataset=self.dataset, directed=True)
        # V, E, Obs, _ = dw.get_data_case(self.hour, self.weekday, self.ref_per)
        with open(self.org_fileName, 'rb') as pkl_file:
            [V, E, Obs, E_X, X_b] = pickle.load(pkl_file)
        ObsO=copy.deepcopy(Obs)
        T = len(Obs[E[0]])

        # E_X = sample_X2(self.test_ratio, V, E)

        """Step 2: Add noise to observations on the edges """
        E_Y = [e for e in E if not E_X.has_key(e)]
        X_b = []
        """ |p_y+alpha*sign(nabla_py L)| <= gamma"""
        if self.gamma > 0.0:
            sign_grad_py = gen_adv_exmaple_csl(V, E, Obs, E_X)
            for e in E_Y:
                # print type(sign_grad_py[0])
                if e not in sign_grad_py[0].keys(): print "Eroorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr!"
                for t in range(0, T):
                    for i in range(len(sign_grad_py[t][e])):
                        Obs[e][t] = clip01(Obs[e][t] + self.alpha * sign_grad_py[t][e][i])  # clip between [0,1]
                    if np.abs(Obs[e][t] - ObsO[e][t]) > self.gamma:
                        Obs[e][t] = clip01(ObsO[e][t] + np.sign(
                            Obs[e][t] - ObsO[e][t]) * self.gamma)  # clip |py_adv-py_orig|<gamma
            print "Iteration Number", [len(sign_grad_py[i][sign_grad_py[i].keys()[0]]) for i in
                                       range(len(sign_grad_py))]

        """   V, E, Obs, Omega, b, X_b, E_X, logging, psl   """

        pkl_file = open(f, 'wb')
        pickle.dump([V, E, Obs, E_X, X_b], pkl_file)
        pkl_file.close()
        # print f

    def __str__(self):
        return '%s' % (self.p0)

"""
Generate simulation datasets with different graph sizes and rates
"""
def traffic_data_generator_rf():
    data_root = "/network/rit/lab/ceashpc/adil/data/adv_csl/Jan2/traffic/random_flip/"


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
    for ref_per in ref_pers[:1]:
        for dataset in datasets[:]:
            dataroot=data_root+dataset+"/"
            if not os.path.exists(dataroot):
                os.makedirs(dataroot)
            for weekday in range(5)[:1]:
                for hour in range(8, 22)[:1]:
                    for test_ratio in [0.1, 0.2, 0.3, 0.4, 0.5][:]:
                        for gamma in [0.0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.20, 0.25][:]:  # 8
                            for real_i in range(realizations)[:]:
                                tasks.put(Task_generate_rf(dataroot,dataset, hour, weekday, ref_per,test_ratio,gamma,real_i))
                                num_jobs += 1
    for i in range(num_consumers):
        tasks.put(None)

    while num_jobs:
        results.get()
        num_jobs -= 1
        print num_jobs


def traffic_data_generator_rn():
    data_root = "/network/rit/lab/ceashpc/adil/data/adv_csl/Jan2/traffic/random_noise/"


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
    for ref_per in ref_pers[:1]:
        for dataset in datasets[:]:
            dataroot=data_root+dataset+"/"
            if not os.path.exists(dataroot):
                os.makedirs(dataroot)
            for weekday in range(5)[:1]:
                for hour in range(8, 22)[:1]:
                    for test_ratio in [0.1, 0.2, 0.3, 0.4, 0.5][:]:
                        for gamma in [0.0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.20, 0.25][:]:  # 8
                            for real_i in range(realizations)[:]:
                                tasks.put(Task_generate_rn(dataroot,dataset, hour, weekday, ref_per,test_ratio,gamma,real_i))
                                num_jobs += 1
    for i in range(num_consumers):
        tasks.put(None)

    while num_jobs:
        results.get()
        num_jobs -= 1
        print num_jobs


def traffic_data_generator_PGD():
    data_root = "/network/rit/lab/ceashpc/adil/data/adv_csl/Jan2/traffic/random_pgd/"


    tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()
    num_consumers = 1  # We only use 5 cores.
    # print 'Creating %d consumers' % num_consumers
    consumers = [Consumer(tasks, results)
                 for i in range(num_consumers)]
    for w in consumers:
        w.start()
    num_jobs=0
    realizations=10
    alpha = 0.01
    ref_pers = [0.6,0.7, 0.8]
    datasets = ['philly', 'dc']
    for ref_per in ref_pers[:1]:
        for dataset in datasets[1:]:
            dataroot=data_root+dataset+"/"
            if not os.path.exists(dataroot):
                os.makedirs(dataroot)
            for weekday in range(5)[:1]:
                for hour in range(8, 22)[:1]:
                    for test_ratio in [0.1, 0.2, 0.3, 0.4, 0.5][:]:
                        for gamma in [0.0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.20, 0.25][:]:  # 8
                            for real_i in range(realizations)[:]:
                                tasks.put(Task_generate_PGD(dataroot,dataset, hour, weekday, ref_per,test_ratio,gamma,real_i,alpha))
                                num_jobs += 1
    for i in range(num_consumers):
        tasks.put(None)

    while num_jobs:
        results.get()
        num_jobs -= 1
        print num_jobs


def traffic_data_generator_PGD_csl():
    data_root = "/network/rit/lab/ceashpc/adil/data/adv_csl/Jan2/traffic/random_pgd_csl/"
    org_data_root="/network/rit/lab/ceashpc/adil/data/adv_csl/Jan2/traffic/random_pgd/"

    tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()
    num_consumers = 1  # We only use 5 cores.
    # print 'Creating %d consumers' % num_consumers
    consumers = [Consumer(tasks, results)
                 for i in range(num_consumers)]
    for w in consumers:
        w.start()
    num_jobs=0
    realizations=10
    alpha = 0.01
    ref_pers = [0.6,0.7, 0.8]
    datasets = ['philly', 'dc']
    for ref_per in ref_pers[:1]:
        for dataset in datasets[1:]:
            dataroot=data_root+dataset+"/"
            if not os.path.exists(dataroot):
                os.makedirs(dataroot)
            for weekday in range(5)[:1]:
                for hour in range(8, 22)[:1]:
                    for test_ratio in [0.1, 0.2, 0.3, 0.4, 0.5][:]:
                        for real_i in range(realizations)[:1]:
                            org_fileName=org_data_root+'/network_{}_weekday_{}_hour_{}_refspeed_{}-testratio-{}-gamma-{}-realization-{}.pkl'.format(
            dataset, weekday,hour,ref_per,test_ratio, 0.0,real_i)
                            for gamma in [0.0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.20, 0.25][:]:  # 8
                                tasks.put(Task_generate_PGD_csl(dataroot,dataset, hour, weekday, ref_per,test_ratio,gamma,real_i,alpha,org_fileName))
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


def gen_adv_exmaple(V, E, Obs, X_b,E_X):
    logging = Log()
    b={}
    Omega = calc_Omega_from_Obs2(Obs, E)                #V, E, Obs, Omega, b, X, logging, psl=False, approx=True, init_alpha_beta=(1, 1),report_stat=False
    sign_grad_py=inference_apdm_format_conflict_evidence(V, E, Obs, Omega, b, E_X, logging, psl=False)


    return sign_grad_py


def gen_adv_exmaple_csl(V, E, Obs,E_X):
    logging = Log()
    b={}
    Omega = calc_Omega_from_Obs2(Obs, E)  #V, E, Obs, Omega, E_X, logging, psl = False,
    sign_grad_py=inference_apdm_format_csl(V, E, Obs, Omega, E_X, logging, psl=False)


    return sign_grad_py

def main():
    # filename = "data/trust-analysis/nodes-47676.pkl"
    # pkl_file = open(filename, 'rb')
    # [V, E] = pickle.load(pkl_file)
    # print len(V), len(E)
    # simulation_data_generator2()
    # simulation_data_generator3()

    # traffic_data_generator_rf()
    # traffic_data_generator_rn()
    # traffic_data_generator_PGD()
    traffic_data_generator_PGD_csl()
if __name__=='__main__':
    main()

