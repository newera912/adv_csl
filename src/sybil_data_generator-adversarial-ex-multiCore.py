import random
import pickle
import os,copy,time,sys
import multiprocessing
from random import shuffle
import numpy as np
from log import Log
from network_funs import *
from gen_sybils_adv_example_structure import inference_apdm_format as inference_structure
from gen_sybils_adv_example import inference_apdm_format as inference_apdm_format_conflict_evidence
from gen_sybils_adv_example_csl import inference_apdm_format as inference_apdm_format_csl

from collections import Counter

def analyze_data_facebook():
    for attack_ratio in [0.1][:]:
        nodes={}
        edges={}
        #8078,88234 4039
        #723244 67392 33696
        #2016920 164336 82168
        TT=4039
        with open("../data/facebook/graph.txt") as op:
            for line in op.readlines():
                e=map(int,line.strip().split())
                nodes[e[0]] = 0 if e[0] < TT else 1
                nodes[e[1]] = 0 if e[1] < TT else 1
                if (e[0]<TT and e[1]<TT): edges[(e[0],e[1])]=0
                elif (e[0] >= TT and e[1] >= TT): edges[(e[0], e[1])] = 1
                # else: edges[(e[0], e[1])] = -1
        print "without attack edges:",len(edges)
        total_edge=len(edges)
        attack_edge=int(attack_ratio*total_edge)
        for i in range(attack_edge):
            e_0=random.choice(range(0,TT))
            e_1=random.choice(range(TT,2*TT))
            while True:
                if not edges.has_key((e_0,e_1)) or  not edges.has_key((e_1,e_0)):
                    break
                e_0 = random.choice(range(0, TT))
                e_1 = random.choice(range(TT, 2*TT))
                attack_edge+=1
                # continue
            edges[(e_0,e_1)]=-1
            edges[(e_1, e_0)] = -1
        # print attack_edge,len(nodes),min(nodes.keys()),max(nodes.keys()),2*4039
        edge_type={}
        for edge in edges.values():
            if edge_type.has_key(edge):
                edge_type[edge]+=1
            else:
                edge_type[edge] = 1
        print edge_type
        print attack_edge,len(edges.keys()),2*TT+2*1000*(attack_edge/1000),2*sum(edges.values())
        attack_edge=1000 * (attack_edge / 1000)
        outfp = open("../data/facebook/facebook_sybils_attackedge_{}.pkl".format(attack_edge), 'w')
        pickle.dump([nodes,edges], outfp)
        outfp.close()

def analyze_data_slashdot():
    for attack_edge in [1000,5000,10000,15000,20000][:]:
        nodes={}
        edges={}
        #8078,88234 4039
        #723244 67392 33696
        #2016920 164336 82168
        TT=82168
        with open("../data/slashdot/slashdot-graph.txt") as op:
            for line in op.readlines():
                e=map(int,line.strip().split())
                nodes[e[0]] = 0 if e[0] < TT else 1
                nodes[e[1]] = 0 if e[1] < TT else 1
                if (e[0]<TT and e[1]<TT): edges[(e[0],e[1])]=0
                elif (e[0] >= TT and e[1] >= TT): edges[(e[0], e[1])] = 1
                # else: edges[(e[0], e[1])] = -1
        print len(edges)
        for i in range(attack_edge):
            e_0=random.choice(range(0,TT))
            e_1=random.choice(range(TT,2*TT))
            while True:
                if not edges.has_key((e_0,e_1)) or  not edges.has_key((e_1,e_0)):
                    break
                e_0 = random.choice(range(0, TT))
                e_1 = random.choice(range(TT, 2*TT))
                attack_edge+=1
                # continue
            edges[(e_0,e_1)]=-1
            edges[(e_1, e_0)] = -1
        # print attack_edge,len(nodes),min(nodes.keys()),max(nodes.keys()),2*4039
        edge_type={}
        for edge in edges.values():
            if edge_type.has_key(edge):
                edge_type[edge]+=1
            else:
                edge_type[edge] = 1
        print edge_type
        print attack_edge,len(edges.keys()),2*TT+2*1000*(attack_edge/1000),2*sum(edges.values())
        attack_edge=1000 * (attack_edge / 1000)
        outfp = open("../data/slashdot/slashdot_sybils_attackedge_{}.pkl".format(attack_edge), 'w')
        pickle.dump([nodes,edges], outfp)
        outfp.close()

def analyze_data():
    for attack_edge in [1000,5000,10000,15000,20000][:]:
        nodes={}
        edges={}
        #8078,88234 4039
        #723244 67392 33696
        TT=33696
        with open("../data/enron/enron-graph.txt") as op:
            for line in op.readlines():
                e=map(int,line.strip().split())
                nodes[e[0]] = 0 if e[0] < TT else 1
                nodes[e[1]] = 0 if e[1] < TT else 1
                if (e[0]<TT and e[1]<TT): edges[(e[0],e[1])]=0
                elif (e[0] >= TT and e[1] >= TT): edges[(e[0], e[1])] = 1
                # else: edges[(e[0], e[1])] = -1
        print len(edges)
        for i in range(attack_edge):
            e_0=random.choice(range(0,TT))
            e_1=random.choice(range(TT,2*TT))
            while True:
                if not edges.has_key((e_0,e_1)) or  not edges.has_key((e_1,e_0)):
                    break
                e_0 = random.choice(range(0, TT))
                e_1 = random.choice(range(TT, 2*TT))
                attack_edge+=1
                # continue
            edges[(e_0,e_1)]=-1
            edges[(e_1, e_0)] = -1
        # print attack_edge,len(nodes),min(nodes.keys()),max(nodes.keys()),2*4039
        edge_type={}
        for edge in edges.values():
            if edge_type.has_key(edge):
                edge_type[edge]+=1
            else:
                edge_type[edge] = 1
        print edge_type
        print attack_edge,len(edges.keys()),723244+2*1000*(attack_edge/1000),2*sum(edges.values())
        attack_edge=1000 * (attack_edge / 1000)
        outfp = open("../data/enron/enron_attackedge_{}.pkl".format(attack_edge), 'w')
        pickle.dump([nodes,edges], outfp)
        outfp.close()


def reformat_edge_in_omega(Omega, node_2_id):
    Omega1 = {}
    for e, alpha_beta in Omega.items():
        Omega1[node_2_id[e]] = alpha_beta
    return Omega1

def reformat(V, E, Obs, Omega = None):
    feat = {}
    id_2_node = {}
    node_2_id = {}
    for idx, v in enumerate(V.keys()):
        id_2_node[idx] = v
        node_2_id[v] = idx
    for v, v_feat in Obs.items():
        feat[node_2_id[v]] = v_feat
    if Omega != None:
        Omega1 = reformat_edge_in_omega(Omega, node_2_id)
    else:
        Omega1 = None

    nns=gen_nns(V,E)
    node_nns = {}
    for v,v_nns in nns.items():
        if not node_nns.has_key(node_2_id[v]):
            node_nns[node_2_id[v]]=[]
            for vv in v_nns:
                node_nns[node_2_id[v]].append(node_2_id[vv])
        else:
            for vv in v_nns:
                node_nns[node_2_id[v]].append(node_2_id[vv])
    # VV={}
    # EE={}
    # for k,v in V.items():
    #     VV[node_2_id[k]]=v
    # for k,v in E.items():
    #     EE[(node_2_id[k[0]],node_2_id[k[1]])]=v




    return node_nns, id_2_node, node_2_id, Omega1, feat

def gen_nns(V, E):
    nns = {v: [] for v in V}
    for (v1,v2) in E:
        if nns.has_key(v1):
            nns[v1].append(v2)
    return nns

"""
Randomly select swap_ratio edges and swap their observations (0 -> 1, 1 -> 1)
"""
def pertubation(obs, swap_ratio):
    for e, feat in obs.items():
        if random.random() < swap_ratio:
            obs[e] = 1 - feat
    return obs


def graph_process(V, E, n_features,swap_ratio):
    obs = {k: v for k,v in V.items()}

    feat_data = {v: [] for v in V.keys()}
    for feat_i in range(n_features):
        if swap_ratio!=0.0:
            obs = pertubation(obs, swap_ratio)
        for (e, feat) in obs.items():
            feat_data[e].append(feat)
        #print "{} # edges: {} # positives: {}".format(feat_i, n_E, np.sum([item[0] for item in feat_data.values()]))
    return feat_data

def sample_X(test_ratio, V):
    pos_nodes=[]
    neg_nodes=[]
    for k,v in V.items():
        if v==0:
            pos_nodes.append(k)  #benign nodes
        elif v==1:
            neg_nodes.append(k)  #sybil nodes
    # print len(neg_nodes),len(pos_nodes),len(V)
    if len(neg_nodes)+len(pos_nodes)!=len(V): print "error........................................."
    N=len(V)
    rands_pos = np.random.permutation(pos_nodes)[:int(np.round(0.5*test_ratio * N))]
    rands_neg = np.random.permutation(neg_nodes)[:int(np.round(0.5 *test_ratio * N))]
    # print rands_neg
    # print rands_pos
    # print list(rands_pos)+list(rands_neg)
    E_X = {i:V[i] for i in list(rands_pos)+list(rands_neg)}
    print test_ratio, len(E_X), len(V), 1.0*len(E_X) / len(V)
    return E_X

def sample_X2(test_ratio, V):
    rand_seq_V = copy.deepcopy(V.keys())
    shuffle(rand_seq_V)
    cnt = int(np.round(len(V) * test_ratio))
    X = rand_seq_V[:cnt]
    E_X = {i:V[i] for i in X}
    print test_ratio, len(E_X), len(V), 1.0*len(E_X) / len(V)
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
    def __init__(self, V, E,Obs, T,test_ratio,swap_ratio,gamma,fout):
        self.V = V
        self.E = E
        self.Obs =Obs
        self.T = T
        self.test_ratio = test_ratio
        self.swap_ratio=swap_ratio
        self.gamma = gamma
        self.fout = fout
    def __call__(self):


        """Step2: Sampling X edges with test ratio """
        # nns, id_2_node, node_2_id, _, _ = reformat(self.V, self.E, Obs)
        # nns=gen_nns(self.V,self.E)
        E_X = sample_X(self.test_ratio, self.V)  #test nodes

        """Step 3: flip observations on the edges """
        E_Y = [v for v in self.V.keys() if not E_X.has_key(v)]
        rand_seq_E_Y = copy.deepcopy(E_Y)
        shuffle(rand_seq_E_Y)
        cnt = int(np.round(len(E_Y) * self.gamma))
        X_b = rand_seq_E_Y[:cnt]
        # T = len(self.Obs[self.E[0]])
        #mid = int(np.round(T / 2.0))
        for v in X_b:
            for t in range(0, self.T):
                self.Obs[v][t] = 1 - self.Obs[v][t]

        pkl_file = open(self.fout, 'wb')
        pickle.dump([self.V, self.E, self.Obs, E_X, X_b], pkl_file)
        pkl_file.close()
        return

    def __str__(self):
        return '%s' % (self.p0)

def gen_adv_exmaple(V, E, Obs, X_b,E_X):

    b={}
    Omega = calc_Omega_from_Obs2(Obs, V)                #V, E, Obs, Omega, b, X_b, E_X, logging, psl=False, approx=True, init_alpha_beta=(1, 1),report_stat=False
    sign_grad_py=inference_apdm_format_conflict_evidence(V, E, Obs, Omega, b, E_X, logging, psl=False)


    return sign_grad_py

def gen_adv_exmaple_structure(V, E, Obs,E_X,v0):
    # logging = Log()
    b={}
    Omega = calc_Omega_from_Obs2(Obs, V)                #V, E, Obs, Omega, b, X_b, E_X, logging, psl=False, approx=True, init_alpha_beta=(1, 1),report_stat=False
    sign_grad_py=inference_structure(V, E, Obs, Omega, b, E_X,v0, None, psl=False)


    return sign_grad_py

def gen_adv_exmaple_csl(V, E, Obs, X_b,E_X):
    logging = Log()
    b={}
    Omega = calc_Omega_from_Obs2(Obs, V)   #V, E, Obs, Omega, E_X, logging,
    sign_grad_py=inference_apdm_format_csl(V, E, Obs, Omega, E_X, logging, psl=False)

    return sign_grad_py

def clip01(x):
    if x<0.0: return 0.0
    elif x>1.0: return 1.0
    else: return x

class Task_generate_rn(object):
    def __init__(self, V, E,Obs, T,test_ratio,swap_ratio,gamma,fout):
        self.V = V
        self.E = E
        self.Obs = Obs
        self.T = T
        self.test_ratio = test_ratio
        self.swap_ratio=swap_ratio
        self.gamma = gamma
        self.fout = fout
    def __call__(self):
        """Step1: Sampling X edges with test ratio """""
        # nns, id_2_node, node_2_id, _, _ = reformat(self.V, self.E, Obs)
        # nns=gen_nns(self.V,self.E)
        E_X = sample_X(self.test_ratio, self.V)  #test nodes

        """Step 2: Add noise to observations on the edges """
        E_Y = [v for v in self.V.keys() if not E_X.has_key(v)]
        X_b = []
        """ |noise_vector_i| <= gamma"""
        noise_vector = np.random.uniform(low=-self.gamma, high=self.gamma, size=(len(E_Y),))

        # T = len(self.Obs[self.E.keys()[0]])
        for i,v in enumerate(E_Y):
            for t in range(0, self.T):
                self.Obs[v][t] = clip01(self.Obs[v][t]+noise_vector[i])   #clip between [0,1]

        pkl_file = open(self.fout, 'wb')
        pickle.dump([self.V, self.E, self.Obs, E_X, X_b], pkl_file)
        pkl_file.close()
        return

    def __str__(self):
        return '%s' % (self.p0)

class Task_generate_pgd(object): #dataset,attack_edge,V, E, Obs, T, test_ratio, swap_ratio,alpha
    def __init__(self, dataset,attack_edge,V, E,Obs, T,test_ratio,swap_ratio,real_i,alpha,out_folder):
        self.dataset =dataset
        self.attack_edge = attack_edge
        self.V = V
        self.E = E
        self.ObsO = Obs
        self.T = T
        self.test_ratio = test_ratio
        self.swap_ratio=swap_ratio
        self.real_i=real_i
        self.alpha = alpha
        self.out_folder = out_folder

    def __call__(self):
        """Step1: Sampling X edges with test ratio """""
        # nns, id_2_node, node_2_id, _, _ = reformat(self.V, self.E, Obs)
        # nns=gen_nns(self.V,self.E)



        """ Step1: Sampling X edges with test ratio """
        E_X = sample_X(self.test_ratio, self.V)  #test nodes

        """Step 2: Add noise to observations on the edges """
        E_Y = [v for v in self.V.keys() if not E_X.has_key(v)]
        X_b = []
        sign_grad_py = gen_adv_exmaple(self.V, self.E, self.ObsO, X_b, E_X)
        print("Get Grad-Sign: finihsed...........")
        """ |p_y+alpha*sign(nabla_py L)| <= gamma"""
        for gamma in [0.0, 0.01, 0.03, 0.05, 0.07,0.09,0.2,0.3,0.4][:]:  # 11
            fout = self.out_folder + "{}-attackedges-{}-T-{}-testratio-{}-swap_ratio-{}-gamma-{}-realization-{}-data-X.pkl".format(
                self.dataset, self.attack_edge, self.T, self.test_ratio, self.swap_ratio, gamma, self.real_i)
            print(fout)
            Obs = copy.deepcopy(self.ObsO)
            if gamma>0.0:

                # T = len(self.Obs[self.E.keys()[0]])
                for e in E_Y:
                    # print type(sign_grad_py[0])
                    if e not in sign_grad_py[0].keys(): print "Eroorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr!"
                    for t in range(0, self.T):
                        for i in range(len(sign_grad_py[t][e])):
                            Obs[e][t] = clip01(Obs[e][t]+self.alpha*sign_grad_py[t][e][i])   #clip between [0,1]
                        if np.abs(Obs[e][t]-self.ObsO[e][t]) >gamma:
                            Obs[e][t]=clip01(self.ObsO[e][t]+np.sign(Obs[e][t]-self.ObsO[e][t])*gamma)  #clip |py_adv-py_orig|<gamma
                print "Iteration Number",[len(sign_grad_py[i][sign_grad_py[i].keys()[0]]) for i in range(len(sign_grad_py)) ]

                # for i in range(10):
                #     for v in sign_grad_py[i].keys():
                #         # print(sign_grad_py[i][v])
                #         if sign_grad_py[i][v][0]==0.0: print v,sign_grad_py[i][v]
                pkl_file = open(fout, 'wb')
                pickle.dump([self.V, self.E, Obs, E_X, X_b], pkl_file)
                pkl_file.close()
            else:
                pkl_file = open(fout, 'wb')
                pickle.dump([self.V, self.E, self.ObsO, E_X, X_b], pkl_file)
                pkl_file.close()


        """   V, E, Obs, Omega, b, X_b, E_X, logging, psl   """


        return

    def __str__(self):
        return '%s' % (self.p0)

class Task_generate_strusture(object): #dataset,attack_edge,V, E, Obs, T, test_ratio, swap_ratio,alpha
    def __init__(self, V, E,Obs,E_X,T,fout,target_nodes,perturbation,adj_list):
        self.V = V
        self.E = E
        self.Obs = Obs
        self.E_X= E_X
        self.T = T
        self.fout = fout
        self.target_nodes=target_nodes
        self.perturbation=perturbation
        self.adj_list = adj_list
    def __call__(self):
        E0 = copy.deepcopy(self.E)
        add_remove=[]
        candidate_edges = []
        for i, v0 in enumerate(self.target_nodes):
            print(">>>>>>>>>>>>>>>>>>>",self.perturbation,i,v0,"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            num_perturbtion = 0
            candidate_edges = []
            if len(self.adj_list[v0]) < self.perturbation:
                for u in self.adj_list[v0][:-1]:
                    E0.pop((v0, u), None)
                    E0.pop((u, v0), None)
                    add_remove.append(-1)
                print(self.perturbation,"[adj<perturbation] |E|:{}, |E0|:{}".format(len(self.E), len(E0)))
                # pkl_file = open(fout, 'wb')
                # pickle.dump([V, E0, Obs, E_X, X_b], pkl_file)
                # pkl_file.close()
                continue
            else:
                for u in self.adj_list[v0]:
                    if self.E_X.has_key(u): continue
                    candidate_edges.extend([(v0, u), (u, v0)])
                print("candidate edges:", len(candidate_edges))
                temp_removed_edges = []
                for e in candidate_edges:
                    if num_perturbtion >= self.perturbation: break
                    if not E0.has_key(e): print(e, "e not exists")
                    if not E0.has_key((e[1], e[0])): print((e[1], e[0]), "e' not exists")
                    if not E0.has_key(e) and not E0.has_key((e[1], e[0])):
                        E0[e] = 1.0
                        E0[(e[1], e[0])] = 1.0
                        add_remove.append(1)
                        print(self.perturbation,num_perturbtion,"[ADD] |E|:{}, |E0|:{}".format(len(self.E), len(E0)))
                        num_perturbtion += 1
                        continue

                    E0.pop(e, None)
                    E0.pop((e[1], e[0]), None)
                    p_adv = gen_adv_exmaple_structure(self.V, E0, self.Obs, self.E_X, v0)

                    if p_adv == self.Obs[v0][0]:  # no changes
                        print(self.perturbation,num_perturbtion,p_adv, self.Obs[v0][0], "not changed...")
                        E0[e] = 1.0
                        E0[(e[1], e[0])] = 1.0
                    else:
                        print(self.perturbation,num_perturbtion,p_adv, self.Obs[v0][0], "changed...")
                        add_remove.append(1)
                        temp_removed_edges.append(e)
                        temp_removed_edges.append((e[1], e[0]))
                        num_perturbtion += 1
                    print(self.perturbation,num_perturbtion,"[DEL ] |E|:{}, |E0|:{}".format(len(self.E), len(E0)))
                if num_perturbtion < self.perturbation:
                    for ee in candidate_edges:
                        if num_perturbtion >= self.perturbation:
                            break
                        if ee not in temp_removed_edges:
                            E0.pop(ee, None)
                            E0.pop((ee[1], ee[0]),None)
                            num_perturbtion += 1
            sys.stdout.flush()
        print(self.perturbation,"[Final output] |E|:{}, |E0|:{} *********************************".format(len(self.E), len(E0)))
        print(self.perturbation,num_perturbtion,add_remove)
        pkl_file = open(self.fout, 'wb')
        pickle.dump([self.V, E0, self.Obs, self.E_X, self.target_nodes], pkl_file)
        pkl_file.close()
        return

    def __str__(self):
        return '%s' % (self.p0)

def generate_structure(dataset,attack_edge,V, E,Obs, T,test_ratio,swap_ratio,real_i,out_folder,node_degree,adj_list):

    """Step1: Sampling X edges with test ratio """""
    # nns, id_2_node, node_2_id, _, _ = reformat(V, E, Obs)
    # nns=gen_nns(V,E)
    """ Step1: Sampling X edges with test ratio """
    E_X = sample_X(test_ratio, V)  #test nodes
    X_keys=list(E_X.keys())
    random.shuffle(X_keys)
    target_nodes={}
    for k in X_keys:
        if len(target_nodes)>=20: break
        if node_degree[k]/2.0>22.5 and node_degree[k]/2.0<=100:
            target_nodes[k]=E_X[k]
            # print k,node_degree[k]/2.0

    print("number of target nodes:",len(target_nodes))
    """Step 2: Add noise to observations on the edges """
    E_Y = [v for v in V.keys() if not E_X.has_key(v)]

    tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()
    num_consumers = 6  # We only use 5 cores.

    print 'Creating %d consumers' % num_consumers
    consumers = [Consumer(tasks, results)
                 for i in range(num_consumers)]
    for w in consumers:
        w.start()

    num_jobs=0
    for perturbation in [0.0,5, 10, 20,30,40, 50,60,70,80,90,100][7:]:  # 11
        fout = out_folder + "{}-attackedges-{}-T-{}-testratio-{}-swap_ratio-{}-perturbation-{}-realization-{}-data-X20.pkl".format(
            dataset, attack_edge, T, test_ratio, swap_ratio, perturbation, real_i)
        print(perturbation,fout)
        # L_curr_value=-float("inf")
        if perturbation==0.0:
            pkl_file = open(fout, 'wb')
            pickle.dump([V, E, Obs, E_X, target_nodes], pkl_file)
            pkl_file.close()
        else:
            tasks.put(Task_generate_strusture(V, E,Obs,E_X,T,fout,target_nodes,perturbation,adj_list))
            num_jobs += 1
    for i in range(num_consumers):
        tasks.put(None)

    while num_jobs:
        results.get()
        num_jobs -= 1
        print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> remain: ", num_jobs

            # E0 = copy.deepcopy(E)
            #
            # candidate_edges=[]
            # for v0 in target_nodes:
            #     num_perturbtion = 0
            #     if len(adj_list[v0])<perturbation:
            #         for u in adj_list[v0][:]:
            #             E0.pop((v0,u),None)
            #             E0.pop((u,v0),None)
            #         print("[adj<perturbation] |E|:{}, |E0|:{}".format(len(E),len(E0)))
            #         # pkl_file = open(fout, 'wb')
            #         # pickle.dump([V, E0, Obs, E_X, X_b], pkl_file)
            #         # pkl_file.close()
            #         continue
            #     else:
            #         for u in adj_list[v0]:
            #             if E_X.has_key((v0,u)): continue
            #             if E_X.has_key((u,v0)): continue
            #             candidate_edges.extend([(v0,u),(u,v0)])
            #         print("candidate edges:",len(candidate_edges))
            #         temp_removed_edges=[]
            #         for e in candidate_edges:
            #             if num_perturbtion>=perturbation: break
            #             if not E0.has_key(e): print(e,"e not exists")
            #             if not E0.has_key((e[1],e[0])): print((e[1],e[0]), "e' not exists")
            #             if not E0.has_key(e) and not E0.has_key((e[1],e[0])):
            #                 E0[e] = E[e]
            #                 E0[(e[1], e[0])] = E[(e[1], e[0])]
            #                 print("[inner ourput>>>] |E|:{}, |E0|:{}".format(len(E), len(E0)))
            #                 num_perturbtion += 1
            #                 continue
            #
            #
            #             E0.pop(e, None)
            #             E0.pop((e[1],e[0]),None)
            #             p_adv = gen_adv_exmaple_structure(V, E0, Obs, E_X,v0)
            #
            #             if p_adv == Obs[v0][0]: #no changes
            #                 print(p_adv,Obs[v0][0],"not changed...")
            #                 E0[e]=E[e]
            #                 E0[(e[1], e[0])]=E[(e[1], e[0])]
            #             else:
            #                 print(p_adv,Obs[v0][0],"changed...")
            #                 temp_removed_edges.append(e)
            #                 temp_removed_edges.append((e[1],e[0]))
            #                 num_perturbtion+=1
            #             print("[inner ourput remove>>>] |E|:{}, |E0|:{}".format(len(E), len(E0)))
            #         if num_perturbtion<perturbation:
            #             for e in candidate_edges:
            #                 if num_perturbtion>=perturbation:
            #                     break
            #                 if e not in temp_removed_edges:
            #                     E0.pop(e, None)
            #                     E0.pop((e[1], e[0]))
            #                     num_perturbtion+=1
            #
            # print("[ourput] |E|:{}, |E0|:{}".format(len(E), len(E0)))
            # pkl_file = open(fout, 'wb')
            # pickle.dump([V, E0, Obs, E_X,target_nodes], pkl_file)
            # pkl_file.close()




    """   V, E, Obs, Omega, b, X_b, E_X, logging, psl   """


    return


class Task_generate_pgd_csl(object): #dataset,attack_edge,V, E, Obs, T, test_ratio, swap_ratio,alpha
    def __init__(self, dataset,attack_edge,V, E,Obs, T,test_ratio,swap_ratio,real_i,alpha,out_folder):
        self.dataset =dataset
        self.attack_edge = attack_edge
        self.V = V
        self.E = E
        self.ObsO = Obs
        self.T = T
        self.test_ratio = test_ratio
        self.swap_ratio=swap_ratio
        self.real_i=real_i
        self.alpha = alpha
        self.out_folder = out_folder

    def __call__(self):
        """Step1: Sampling X edges with test ratio """""
        # nns, id_2_node, node_2_id, _, _ = reformat(self.V, self.E, Obs)
        # nns=gen_nns(self.V,self.E)



        """ Step1: Sampling X edges with test ratio """
        E_X = sample_X(self.test_ratio, self.V)  #test nodes

        """Step 2: Add noise to observations on the edges """
        E_Y = [v for v in self.V.keys() if not E_X.has_key(v)]
        X_b = []
        sign_grad_py = gen_adv_exmaple_csl(self.V, self.E, self.ObsO, X_b, E_X)
        """ |p_y+alpha*sign(nabla_py L)| <= gamma"""
        for gamma in [0.0, 0.01, 0.03, 0.05, 0.07,0.09,0.2,0.3,0.4,0.5][:]:  # 11
            fout = self.out_folder + "{}-attackedges-{}-T-{}-testratio-{}-swap_ratio-{}-gamma-{}-realization-{}-data-X.pkl".format(
                self.dataset, self.attack_edge, self.T, self.test_ratio, self.swap_ratio, gamma, self.real_i)
            Obs = copy.deepcopy(self.ObsO)
            if gamma>0.0:
                dir_len = int(round(gamma / self.alpha))
                # T = len(self.Obs[self.E.keys()[0]])
                for e in E_Y:
                    # print type(sign_grad_py[0])
                    if e not in sign_grad_py[0].keys(): print "Eroorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr!"
                    for t in range(0, self.T):
                        delta = {}
                        for i in range(len(sign_grad_py[t][e]))[:dir_len]:
                            tupl=(self.ObsO[e][t],Counter(sign_grad_py[t][e][:dir_len]).most_common(1)[0][0])
                            if delta.has_key(tupl):
                                delta[tupl]+=1
                            else:
                                delta[tupl]=1
                            # print self.ObsO[e][t],Counter(sign_grad_py[t][e]).most_common(1)[0][0]
                            Obs[e][t] = clip01(Obs[e][t]+gamma*sign_grad_py[t][e][i])   #clip between [0,1]
                        if np.abs(Obs[e][t]-self.ObsO[e][t]) >gamma:
                            Obs[e][t]=clip01(self.ObsO[e][t]+np.sign(Obs[e][t]-self.ObsO[e][t])*gamma)  #clip |py_adv-py_orig|<gamma
                        # print delta,sign_grad_py[t][e][:dir_len]
                print "Iteration Number",[len(sign_grad_py[i][sign_grad_py[i].keys()[0]]) for i in range(len(sign_grad_py)) ]
                pkl_file = open(fout, 'wb')
                pickle.dump([self.V, self.E, Obs, E_X, X_b], pkl_file)
                pkl_file.close()
            else:
                pkl_file = open(fout, 'wb')
                pickle.dump([self.V, self.E, self.ObsO, E_X, X_b], pkl_file)
                pkl_file.close()
                print


        """   V, E, Obs, Omega, b, X_b, E_X, logging, psl   """


        return

    def __str__(self):
        return '%s' % (self.p0)




def random_flip_sybils_data_generator():
    data_root = "/network/rit/lab/ceashpc/adil/data/adv_csl/Jan2/random_flip/"
    # data_root = "./"
    realizations = 1
    attack_edges = [1000,2000,3000,4000,5000]
    network_files=["../data/Undirected_Facebook/facebook_sybils_attackedge",
                   "../data/enron/enron_attackedge",
                   "../data/slashdot/slashdot_sybils_attackedge"][2:]
    tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()
    num_consumers = 30  # We only use 5 cores.
    # print 'Creating %d consumers' % num_consumers
    consumers = [Consumer(tasks, results)
                 for i in range(num_consumers)]
    for w in consumers:
        w.start()
    num_jobs=0
    for i,dataset in enumerate(["facebook","enron","slashdot"][2:]):
        for attack_edge in [1000, 5000,10000,15000,20000][:]:
            filename = network_files[i]+"_{}.pkl".format(attack_edge)
            print "--------- reading {}".format(filename)
            pkl_file = open(filename, 'rb')
            [V, E] = pickle.load(pkl_file)
            pkl_file.close()
            out_folder = data_root+dataset+"/"
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            for T in [10][:]:
                for swap_ratio in [0.00, 0.01,0.02, 0.05][1:2]:
                    for test_ratio in [0.1, 0.2, 0.3, 0.4,0.5][:]:
                        """ Step 1: generate observation matrix """
                        Obs = graph_process(V, E, T, swap_ratio)
                        for gamma in [0.0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.20, 0.25][:]:  # 11
                            for real_i in range(realizations)[:]:
                                fout= out_folder+"{}-attackedges-{}-T-{}-testratio-{}-swap_ratio-{}-gamma-{}-realization-{}-data-X.pkl".format(dataset,attack_edge, T, test_ratio,swap_ratio, gamma, real_i)
                                if not os.path.exists(fout):
                                   print fout
                                   tasks.put(Task_generate_rf(V, E,Obs, T,test_ratio,swap_ratio,gamma,fout))
                                   num_jobs+=1
            print "\n\nTttack_edge size {} Done.....................\n\n".format(attack_edge)

        for i in range(num_consumers):
            tasks.put(None)

        while num_jobs:
            results.get()
            num_jobs -= 1
            print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> remain: ",num_jobs


def random_noise_sybils_data_generator():
    data_root = "/network/rit/lab/ceashpc/adil/data/adv_csl/Jan2/random_noise/"
    # data_root = "./"
    realizations = 10
    network_files = {"facebook": "../data/facebook/facebook_sybils_attackedge",
                     "enron": "../data/enron/enron_attackedge",
                     "slashdot": "../data/slashdot/slashdot_sybils_attackedge"}
    tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()
    num_consumers = 30  # We only use 5 cores.
    # print 'Creating %d consumers' % num_consumers
    consumers = [Consumer(tasks, results)
                 for i in range(num_consumers)]
    for w in consumers:
        w.start()
    num_jobs=0
    for i,dataset in enumerate(["facebook","enron","slashdot"][2:]):
        for attack_edge in [1000, 5000,10000,15000,20000,35000][2:3]:
            filename = network_files[dataset] + "_{}.pkl".format(attack_edge)

            pkl_file = open(filename, 'rb')
            [V, E] = pickle.load(pkl_file)
            print "--------- reading {} V:{} E:{}".format(filename,len(V),len(E))
            pkl_file.close()
            out_folder = data_root+dataset+"/"
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            for T in [10][:]:
                for swap_ratio in [0.00, 0.01,0.02, 0.05][1:2]:
                    for test_ratio in [0.3,0.1, 0.2, 0.4,0.5][:1]:
                        Obs = graph_process(V, E, T, swap_ratio)
                        for gamma in [0.0, 0.01, 0.03, 0.05, 0.07,0.09,0.2,0.3,0.4,0.5][:]:  # 11
                            for real_i in range(realizations)[:]:
                                fout = out_folder + "{}-attackedges-{}-T-{}-testratio-{}-swap_ratio-{}-gamma-{}-realization-{}-data-X.pkl".format(
                                    dataset, attack_edge, T, test_ratio, swap_ratio, gamma, real_i)
                                tasks.put(Task_generate_rn(V, E, Obs, T, test_ratio, swap_ratio, gamma, fout))
                                num_jobs += 1
                    print "\n\nAttack_edge size {} Done.....................\n\n".format(attack_edge)

        for i in range(num_consumers):
            tasks.put(None)

        while num_jobs:
            results.get()
            num_jobs -= 1
            print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> remain: ",num_jobs


def pgd_sybils_data_generator():
    data_root = "/network/rit/lab/ceashpc/adil/data/adv_csl/Jan2/random_pgd/"
    org_data_root="/network/rit/lab/ceashpc/adil/data/adv_csl/Jan2/random_pgd_csl/"
    # data_root = "./"
    realizations = 10
    attack_edges = [1000,2000,3000,4000,5000]
    network_files={"facebook":"../data/facebook/facebook_sybils_attackedge",
                   "enron":"../data/enron/enron_attackedge",
                   "slashdot":"../data/slashdot/slashdot_sybils_attackedge"}

    alpha=0.02
    tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()
    num_consumers = 10  # We only use 5 cores.
    # print 'Creating %d consumers' % num_consumers
    consumers = [Consumer(tasks, results)
                 for i in range(num_consumers)]
    for w in consumers:
        w.start()
    num_jobs=0
    for i,dataset in enumerate(["facebook","enron","slashdot"][2:]):
        for attack_edge in [1000, 5000,10000,15000,20000][2:3]:
            filename = network_files[dataset]+"_{}.pkl".format(attack_edge)
            print "--------- reading {}".format(filename)
            pkl_file = open(filename, 'rb')
            [V, E] = pickle.load(pkl_file)
            pkl_file.close()
            out_folder = data_root+dataset+"/"
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            for T in [10][:]:
                for swap_ratio in [0.00, 0.01,0.02, 0.05][1:2]:
                    for test_ratio in [0.3,0.1, 0.2, 0.4,0.5][:1]:
                        for real_i in range(realizations)[:]:
                            Obs = graph_process(V, E, T, swap_ratio)
                            tasks.put(Task_generate_pgd(dataset,attack_edge,V, E, Obs, T, test_ratio, swap_ratio,real_i,alpha,out_folder))
                            num_jobs += 1
                    print "\n\nTttack_edge size {} Done.....................\n\n".format(attack_edge)

        for i in range(num_consumers):
            tasks.put(None)

        while num_jobs:
            results.get()
            num_jobs -= 1
            print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> remain: ",num_jobs

def pgd_csl_sybils_data_generator():
    data_root = "/network/rit/lab/ceashpc/adil/data/adv_csl/Jan2/random_pgd_csl/"
    org_data_root = "/network/rit/lab/ceashpc/adil/data/adv_csl/Jan2/random_pgd_csl/"
    # data_root = "./"
    realizations = 10
    network_files = {"facebook": "../data/facebook/facebook_sybils_attackedge",
                     "enron": "../data/enron/enron_attackedge",
                     "slashdot": "../data/slashdot/slashdot_sybils_attackedge"}

    alpha = 0.02
    tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()
    num_consumers = 10  # We only use 5 cores.
    # print 'Creating %d consumers' % num_consumers
    consumers = [Consumer(tasks, results)
                 for i in range(num_consumers)]
    for w in consumers:
        w.start()
    num_jobs = 0
    for i, dataset in enumerate(["facebook", "enron", "slashdot"][2:]):
        for attack_edge in [1000, 5000, 10000, 15000, 20000][2:3]:
            filename = network_files[dataset] + "_{}.pkl".format(attack_edge)
            print "--------- reading {}".format(filename)
            pkl_file = open(filename, 'rb')
            [V, E] = pickle.load(pkl_file)
            pkl_file.close()
            out_folder = data_root + dataset + "/"
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            for T in [10][:]:
                for swap_ratio in [0.00, 0.01, 0.02, 0.05][1:2]:
                    for test_ratio in [0.3,0.1, 0.2,  0.4, 0.5][:1]:
                        for real_i in range(realizations)[:1]:
                            Obs = graph_process(V, E, T, swap_ratio)
                            tasks.put(
                                Task_generate_pgd_csl(dataset, attack_edge, V, E, Obs, T, test_ratio, swap_ratio, real_i,
                                                  alpha, out_folder))
                            num_jobs += 1
                    print "\n\nAtttack_edge size {} Done.....................\n\n".format(attack_edge)

        for i in range(num_consumers):
            tasks.put(None)

        while num_jobs:
            results.get()
            num_jobs -= 1
            print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> remain: ", num_jobs


def structure_sybils_data_generator():
    data_root =    "/network/rit/lab/ceashpc/adil/data/adv_csl/Jan2/structure/"
    org_data_root="/network/rit/lab/ceashpc/adil/data/adv_csl/Jan2/random_pgd_csl/"
    # data_root="./"
    data_root = "./"
    realizations = 1
    network_files={"facebook":"../data/facebook/facebook_sybils_attackedge",
                   "enron":"../data/enron/enron_attackedge",
                   "slashdot":"../data/slashdot/slashdot_sybils_attackedge"}

    alpha=0.02

    num_jobs=0
    node_degree={}
    adj_list={}
    for i,dataset in enumerate(["facebook","enron","slashdot"][:1]):
        for attack_edge in [1000, 5000,10000,15000,20000][2:3]:
            filename = network_files[dataset]+"_{}.pkl".format(attack_edge)
            print "--------- reading {}".format(filename)
            pkl_file = open(filename, 'rb')
            [V, E] = pickle.load(pkl_file)
            # logging = Log()
            from log import Log
            pkl_file.close()
            for e in E:
                if node_degree.has_key(e[0]):
                    node_degree[e[0]]+=1
                else:
                    node_degree[e[0]] = 1
                if node_degree.has_key(e[1]):
                    node_degree[e[1]]+=1
                else:
                    node_degree[e[1]] = 1

                if adj_list.has_key(e[0]):
                    adj_list[e[0]].append(e[1])
                else:
                    adj_list[e[0]] = [e[1]]
            # print len(node_degree)
            # print len(adj_list),adj_list.values()[:1]
            import collections
            # deg=node_degree.values()
            # deg_dict={}
            # for v in deg:
            #     if deg_dict.has_key(v):
            #         deg_dict[v]+=1
            #     else:
            #         deg_dict[v] = 1
            #
            # for i in sorted(deg_dict):
            #     print ((i/2, deg_dict[i]))
            out_folder = data_root+dataset+"/"
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            for T in [10][:]:
                for swap_ratio in [0.00, 0.01,0.02, 0.05][1:2]:
                    for test_ratio in [0.3,0.1, 0.2, 0.4,0.5][:1]:
                        for real_i in range(realizations)[:]:
                            Obs = graph_process(V, E, T, swap_ratio)
                            t0=time.time()
                            generate_structure(dataset,attack_edge,V, E, Obs, T, test_ratio, swap_ratio,real_i,out_folder,node_degree,adj_list)
                            num_jobs += 1
                            print("running time...",time.time()-t0)
                    print "\n\nTttack_edge size {} Done.....................\n\n".format(attack_edge)




if __name__=='__main__':
    # analyze_data_FB()
    # analyze_data_facebook()

    # random_flip_sybils_data_generator()
    # random_noise_sybils_data_generator()
    # pgd_sybils_data_generator()
    # pgd_csl_sybils_data_generator()
    structure_sybils_data_generator()