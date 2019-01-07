__author__ = 'Adil Alim'
import pickle
from math import *
import numpy as np
import copy
import random
import networkx as nx
from scipy.stats import beta
from cubic import cubic
from basic_funs import *
from network_funs import *
import os
import multiprocessing
from random import shuffle
from gen_epinion_adv_example import inference_apdm_format as inference_apdm_format_conflict_evidence
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
    print test_ratio, len(E_X), len(E), len(E_X) / len(E)
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
    print test_ratio,len(E_X),len(E),count/len(E)
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
    def __init__(self, V, E,Obs, T, swap_ratio,test_ratio,ratio,gamma,fout):
        self.V = V
        self.E = E
        self.ObsO =Obs
        self.T = T
        self.swap_ratio = swap_ratio
        self.test_ratio = test_ratio
        self.ratio = ratio
        self.gamma = gamma
        self.fout = fout
    def __call__(self):

        """ Step 1: generate observation matrix """
        # Obs = graph_process(self.V, self.E, self.T, self.ratio, self.swap_ratio)
        # pos=np.zeros(self.T)
        # for t in range(self.T):
        #     for e in Obs.keys():
        #         pos[t]+=Obs[e][t]
        # if len(Obs)!=len(self.E): print "erorrrr......"
        # pos=pos/len(Obs.keys())
        # print self.T,self.ratio,round(np.mean(pos),2),pos

        Obs=copy.deepcopy(self.ObsO)
        """ Step2: Sampling X edges with test ratio """
        _, edge_down_nns, id_2_edge, edge_2_id, _, _ = reformat(self.V, self.E, Obs)
        _, dict_paths = generate_eopinion_PSL_rules_from_edge_cnns(edge_down_nns, id_2_edge, edge_2_id)
        E_X = sample_X2(self.test_ratio, self.V, [id_2_edge[item] for item in dict_paths.keys()])

        """Step 3: flip observations on the edges """
        E_Y = [e for e in self.E if not E_X.has_key(e)]
        rand_seq_E_Y = copy.deepcopy(E_Y)
        shuffle(rand_seq_E_Y)
        cnt = int(np.round(len(E_Y) * self.gamma))
        X_b = rand_seq_E_Y[:cnt]
        T = len(Obs[self.E[0]])
        #mid = int(np.round(T / 2.0))
        for e in X_b:
            for t in range(0, T):
                Obs[e][t] = 1 - Obs[e][t]

        pkl_file = open(self.fout, 'wb')
        pickle.dump([self.V, self.E, Obs, E_X, X_b], pkl_file)
        pkl_file.close()
        return

    def __str__(self):
        return '%s' % (self.p0)


def clip01(x):
    if x<0.0: return 0.0
    elif x>1.0: return 1.0
    else: return x

class Task_generate_rn(object):
    def __init__(self, V, E,Obs, T, swap_ratio,test_ratio,ratio,gamma,fout):
        self.V = V
        self.E = E
        self.ObsO =Obs
        self.T = T
        self.swap_ratio = swap_ratio
        self.test_ratio = test_ratio
        self.ratio = ratio
        self.gamma = gamma
        self.fout = fout
    def __call__(self):

        """ Step 1: generate observation matrix """
        # Obs = graph_process(self.V, self.E, self.T, self.ratio, self.swap_ratio)
        # pos=np.zeros(self.T)
        # for t in range(self.T):
        #     for e in Obs.keys():
        #         pos[t]+=Obs[e][t]
        # if len(Obs)!=len(self.E): print "erorrrr......"
        # pos=pos/len(Obs.keys())
        # print self.T,self.ratio,round(np.mean(pos),2),pos


        Obs=copy.deepcopy(self.ObsO)
        """ Step1: Sampling X edges with test ratio """
        _, edge_down_nns, id_2_edge, edge_2_id, _, _ = reformat(self.V, self.E, Obs)
        _, dict_paths = generate_eopinion_PSL_rules_from_edge_cnns(edge_down_nns, id_2_edge, edge_2_id)
        E_X = sample_X2(self.test_ratio, self.V, [id_2_edge[item] for item in dict_paths.keys()])

        """Step 2: Add noise to observations on the edges """
        E_Y = [e for e in self.E if not E_X.has_key(e)]

        X_b = []
        """ |noise_vector_i| <= gamma"""
        noise_vector = np.random.uniform(low=-self.gamma, high=self.gamma, size=(len(E_Y),))

        T = len(Obs[self.E[0]])
        for i,e in enumerate(E_Y):
            for t in range(0, T):
                Obs[e][t] = clip01(Obs[e][t]+noise_vector[i])   #clip between [0,1]

        pkl_file = open(self.fout, 'wb')
        pickle.dump([self.V, self.E, Obs, E_X, X_b], pkl_file)
        pkl_file.close()
        return

    def __str__(self):
        return '%s' % (self.p0)


class Task_generate_PGD(object):
    def __init__(self, V, E,Obs, T, swap_ratio,test_ratio,ratio,gamma,alpha,fout):
        self.V = V
        self.E = E
        self.ObsO =Obs
        self.T = T
        self.swap_ratio = swap_ratio
        self.test_ratio = test_ratio
        self.ratio = ratio
        self.gamma = gamma
        self.alpha=alpha
        self.fout = fout
    def __call__(self):

        """ Step 1: generate observation matrix """
        # Obs = graph_process(self.V, self.E, self.T, self.ratio, self.swap_ratio)
        # pos=np.zeros(self.T)
        # for t in range(self.T):
        #     for e in Obs.keys():
        #         pos[t]+=Obs[e][t]
        # if len(Obs)!=len(self.E): print "erorrrr......"
        # pos=pos/len(Obs.keys())
        # print self.T,self.ratio,round(np.mean(pos),2),pos



        Obs=copy.deepcopy(self.ObsO)
        """ Step1: Sampling X edges with test ratio """
        _, edge_down_nns, id_2_edge, edge_2_id, _, _ = reformat(self.V, self.E, Obs)
        _, dict_paths = generate_eopinion_PSL_rules_from_edge_cnns(edge_down_nns, id_2_edge, edge_2_id)
        E_X = sample_X2(self.test_ratio, self.V, [id_2_edge[item] for item in dict_paths.keys()])

        """Step 2: Add noise to observations on the edges """
        E_Y = [e for e in self.E if not E_X.has_key(e)]
        # print E_Y
        # rand_seq_E_Y = copy.deepcopy(E_Y)
        # shuffle(rand_seq_E_Y)
        # cnt = int(np.round(len(E_Y) * self.gamma))
        # X_b = rand_seq_E_Y[:cnt]
        X_b=[]
        """ |p_y+alpha*sign(nabla_py L)| <= gamma"""
        if self.gamma>0.0:
            sign_grad_py = gen_adv_exmaple(self.V, self.E, Obs, X_b, E_X)

            T = len(Obs[self.E[0]])
            for e in E_Y:
                # print type(sign_grad_py[0])
                if e not in sign_grad_py[0].keys(): print "Eroorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr!"
                for t in range(0, self.T):
                    for i in range(len(sign_grad_py[t][e])):
                        Obs[e][t] = clip01(Obs[e][t]+self.alpha*sign_grad_py[t][e][i])   #clip between [0,1]
                    if np.abs(Obs[e][t]-self.ObsO[e][t]) >self.gamma:
                        Obs[e][t]=clip01(self.ObsO[e][t]+np.sign(Obs[e][t]-self.ObsO[e][t])*self.gamma)  #clip |py_adv-py_orig|<gamma
            print "Iteration Number",[len(sign_grad_py[i][sign_grad_py[i].keys()[0]]) for i in range(len(sign_grad_py)) ]


        """   V, E, Obs, Omega, b, X_b, E_X, logging, psl   """

        pkl_file = open(self.fout, 'wb')
        pickle.dump([self.V, self.E, Obs, E_X, X_b], pkl_file)
        pkl_file.close()
        return

    def __str__(self):
        return '%s' % (self.p0)

"""
Generate simulation datasets with different graph sizes and rates
gamma =[0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05,
                                          0.055, 0.06, 0.065, 0.07]
"""


def simulation_data_generator_rf():
    data_root="/network/rit/lab/ceashpc/adil/data/adv_csl/Jan2/random_flip/"
    realizations = 10
    graph_sizes = [1000,5000, 10000,47676]
    # graph_sizes = [2500, 7500]
    ratios = [0.2]

    tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()
    num_consumers = 50  # We only use 5 cores.
    # print 'Creating %d consumers' % num_consumers
    consumers = [Consumer(tasks, results)
                 for i in range(num_consumers)]
    for w in consumers:
        w.start()
    num_jobs=0
    for graph_size in graph_sizes[1:2]:
        filename = "../data/graph_data/nodes-{0}.pkl".format(graph_size)
        print "--------- reading {}".format(filename)
        pkl_file = open(filename, 'rb')
        [V, E] = pickle.load(pkl_file)
        pkl_file.close()
        out_folder = data_root + str(graph_size) + "/"
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        for T in [8,9,10,11][2:3]:
            for swap_ratio in [0.00, 0.01, 0.05][:1]:
                for test_ratio in [0.1, 0.2, 0.3, 0.4,0.5][:]:
                    for ratio in ratios[:1]:  #the percentage of edges set the observations to 1
                        for real_i in range(realizations)[:]:
                            Obs = graph_process(V,E, T, ratio,swap_ratio)
                            for gamma in [0.0, 0.01, 0.03, 0.05, 0.07,0.09,0.11,0.13,0.15,0.20,0.25][:]: #8
                                fout= out_folder+"nodes-{}-T-{}-rate-{}-testratio-{}-swaprate-{}-gamma-{}-realization-{}-data-X.pkl".format(graph_size, T, ratio, test_ratio, swap_ratio, gamma, real_i)
                                if not os.path.exists(fout) or True:
                                   tasks.put(Task_generate_rf(V, E, Obs,T, swap_ratio,test_ratio,ratio,gamma,fout))
                                   num_jobs+=1
        print "\n\nGraph size {} Done.....................\n\n".format(graph_size)

    for i in range(num_consumers):
        tasks.put(None)

    while num_jobs:
        results.get()
        num_jobs -= 1
        print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> remain: ",num_jobs


"""
Generate simulation datasets with different graph sizes and rates
"""
def simulation_data_generator_rn():
    data_root="/network/rit/lab/ceashpc/adil/data/adv_csl/Jan2/random_noise/"
    realizations = 10
    graph_sizes = [1000,5000, 10000,47676]
    # graph_sizes = [2500, 7500]
    ratios = [0.2]

    tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()
    num_consumers = 50  # We only use 5 cores.
    # print 'Creating %d consumers' % num_consumers
    consumers = [Consumer(tasks, results)
                 for i in range(num_consumers)]
    for w in consumers:
        w.start()
    num_jobs=0
    for graph_size in graph_sizes[1:2]:
        filename = "../data/graph_data/nodes-{0}.pkl".format(graph_size)
        print "--------- reading {}".format(filename)
        pkl_file = open(filename, 'rb')
        [V, E] = pickle.load(pkl_file)
        pkl_file.close()
        out_folder = data_root + str(graph_size) + "/"
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        for T in [8,9,10,11][2:3]:
            for swap_ratio in [0.00, 0.01, 0.05][:1]:
                for test_ratio in [0.1, 0.2, 0.3, 0.4,0.5][:]:
                    for ratio in ratios[:]:  #the percentage of edges set the observations to 1
                        for real_i in range(realizations)[:]:
                            Obs = graph_process(V,E, T, ratio,swap_ratio)
                            for gamma in [0.0, 0.01, 0.03, 0.05, 0.07,0.09,0.11,0.13,0.15,0.20,0.25][:]: #8
                                fout= out_folder+"nodes-{}-T-{}-rate-{}-testratio-{}-swaprate-{}-gamma-{}-realization-{}-data-X.pkl".format(graph_size, T, ratio, test_ratio, swap_ratio, gamma, real_i)
                                if not os.path.exists(fout) or True:
                                   tasks.put(Task_generate_rn(V, E, Obs,T, swap_ratio,test_ratio,ratio,gamma,fout))
                                   num_jobs+=1
        print "\n\nGraph size {} Done.....................\n\n".format(graph_size)

    for i in range(num_consumers):
        tasks.put(None)

    while num_jobs:
        results.get()
        num_jobs -= 1
        print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> remain: ",num_jobs

def simulation_data_generator_PGD():
    data_root="/network/rit/lab/ceashpc/adil/data/adv_csl/Jan2/random_pgd/"
    realizations = 10
    graph_sizes = [1000,5000, 10000,47676]
    # graph_sizes = [2500, 7500]
    ratios = [0.2]
    alpha=0.01
    tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()
    num_consumers = 1  # We only use 5 cores.
    # print 'Creating %d consumers' % num_consumers
    consumers = [Consumer(tasks, results)
                 for i in range(num_consumers)]
    for w in consumers:
        w.start()
    num_jobs=0
    for graph_size in graph_sizes[1:2]:
        filename = "../data/graph_data/nodes-{0}.pkl".format(graph_size)
        print "--------- reading {}".format(filename)
        pkl_file = open(filename, 'rb')
        [V, E] = pickle.load(pkl_file)
        pkl_file.close()
        out_folder = data_root + str(graph_size) + "/"
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        for T in [8,9,10,11][2:3]:
            for swap_ratio in [0.00, 0.01, 0.05][:1]:
                for test_ratio in [0.1, 0.2, 0.3, 0.4,0.5][4:]:
                    for ratio in ratios[:]:  #the percentage of edges set the observations to 1
                        for real_i in range(realizations)[:]:
                            Obs = graph_process(V,E, T, ratio,swap_ratio)
                            for gamma in [0.0, 0.01, 0.03, 0.05, 0.07,0.09,0.11,0.13,0.15,0.20,0.25][:]: #11
                                fout= out_folder+"nodes-{}-T-{}-rate-{}-testratio-{}-swaprate-{}-gamma-{}-realization-{}-data-X.pkl".format(graph_size, T, ratio, test_ratio, swap_ratio, gamma, real_i)
                                if not os.path.exists(fout):
                                   tasks.put(Task_generate_PGD(V, E, Obs,T, swap_ratio,test_ratio,ratio,gamma,alpha,fout))
                                   num_jobs+=1
        print "\n\nGraph size {} Done.....................\n\n".format(graph_size)

    for i in range(num_consumers):
        tasks.put(None)

    while num_jobs:
        results.get()
        num_jobs -= 1
        print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> remain: ",num_jobs

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
        #print "{} # edges: {} # positives: {}".format(feat_i, n_E, np.sum([item[0] for item in feat_data.values()]))
    return feat_data

                    #self.V, self.E, Obs, X_b, E_X
def gen_adv_exmaple(V, E, Obs, X_b,E_X):
    logging = Log()
    b={}
    Omega = calc_Omega_from_Obs2(Obs, E)                #V, E, Obs, Omega, b, X_b, E_X, logging, psl=False, approx=True, init_alpha_beta=(1, 1),report_stat=False
    sign_grad_py=inference_apdm_format_conflict_evidence(V, E, Obs, Omega,  b, X_b, E_X, logging, psl=False)


    return sign_grad_py

# def draw_graph(V, E, Obs):
#     G = nx.Graph()
#     G.add_nodes_from(V)
#     for e in E:
#         G.add_edge(e[0],e[1],weight = Obs[e][0])
#     nx.draw(G)
#     nx.draw(G,pos=nx.spectral_layout(G), nodecolor='r',edge_color='b')
#     plt.show()
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
    # simulation_data_generator2()
    simulation_data_generator_rf()
    simulation_data_generator_rn()
    # simulation_data_generator_PGD()

if __name__=='__main__':
    main()

