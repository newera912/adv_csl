import random
import pickle
import os,copy
import multiprocessing
from random import shuffle
import numpy as np

def analyze_data_FB():
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
        nns=gen_nns(self.V,self.E)
        E_X = sample_X(self.test_ratio, self.V)  #test nodes

        """Step 3: flip observations on the edges """
        E_Y = [e for e in self.E if not E_X.has_key(e)]
        rand_seq_E_Y = copy.deepcopy(E_Y)
        shuffle(rand_seq_E_Y)
        cnt = int(np.round(len(E_Y) * self.gamma))
        X_b = rand_seq_E_Y[:cnt]
        T = len(self.Obs[self.E[0]])
        #mid = int(np.round(T / 2.0))
        for e in X_b:
            for t in range(0, T):
                self.Obs[e][t] = 1 - self.Obs[e][t]

        pkl_file = open(self.fout, 'wb')
        pickle.dump([self.V, self.E, self.Obs, E_X, X_b], pkl_file)
        pkl_file.close()
        return

    def __str__(self):
        return '%s' % (self.p0)

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
        E_Y = [e for e in self.E if not E_X.has_key(e)]
        X_b = []
        """ |noise_vector_i| <= gamma"""
        noise_vector = np.random.uniform(low=-self.gamma, high=self.gamma, size=(len(E_Y),))

        T = len(self.Obs[self.E[0]])
        for i,e in enumerate(E_Y):
            for t in range(0, T):
                self.Obs[e][t] = clip01(self.Obs[e][t]+noise_vector[i])   #clip between [0,1]

        pkl_file = open(self.fout, 'wb')
        pickle.dump([self.V, self.E, self.Obs, E_X, X_b], pkl_file)
        pkl_file.close()
        return

    def __str__(self):
        return '%s' % (self.p0)

class Task_generate_rn(object):
    def __init__(self, V, E,Obs, T,test_ratio,swap_ratio,gamma,fout):
        self.V = V
        self.E = E
        self.ObsO = Obs
        self.T = T
        self.test_ratio = test_ratio
        self.swap_ratio=swap_ratio
        self.gamma = gamma
        self.fout = fout
    def __call__(self):
        """Step1: Sampling X edges with test ratio """""
        # nns, id_2_node, node_2_id, _, _ = reformat(self.V, self.E, Obs)
        # nns=gen_nns(self.V,self.E)

        Obs = copy.deepcopy(self.ObsO)

        """ Step1: Sampling X edges with test ratio """
        E_X = sample_X(self.test_ratio, self.V)  #test nodes

        """Step 2: Add noise to observations on the edges """
        E_Y = [e for e in self.E if not E_X.has_key(e)]
        X_b = []
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



def random_flip_sybils_data_generator():
    data_root = "/network/rit/lab/ceashpc/adil/data/adv_csl/Jan2/random_flip/"
    # data_root = "./"
    realizations = 1
    attack_edges = [1000,2000,3000,4000,5000]
    network_files=["../data/Undirected_Facebook/facebook_sybils_attackedge",
                   "../data/enron/enron_attackedge",
                   "../data/slashdot/slashdot_sybils_attackedge"]
    tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()
    num_consumers = 50  # We only use 5 cores.
    # print 'Creating %d consumers' % num_consumers
    consumers = [Consumer(tasks, results)
                 for i in range(num_consumers)]
    for w in consumers:
        w.start()
    num_jobs=0
    for i,dataset in enumerate(["facebook","enron","slashdot"]):
        for attack_edge in [1000, 5000,10000,15000,20000][:]:
            filename = network_files[i]+"{}.pkl".format(attack_edge)
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
    realizations = 1
    attack_edges = [1000,2000,3000,4000,5000]
    network_files=["../data/Undirected_Facebook/facebook_sybils_attackedge",
                   "../data/enron/enron_attackedge",
                   "../data/slashdot/slashdot_sybils_attackedge"]
    tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()
    num_consumers = 50  # We only use 5 cores.
    # print 'Creating %d consumers' % num_consumers
    consumers = [Consumer(tasks, results)
                 for i in range(num_consumers)]
    for w in consumers:
        w.start()
    num_jobs=0
    for i,dataset in enumerate(["facebook","enron","slashdot"][:]):
        for attack_edge in [1000, 5000,10000,15000,20000][:]:
            filename = network_files[i]+"{}.pkl".format(attack_edge)
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
                        Obs = graph_process(V, E, T, swap_ratio)
                        for gamma in [0.0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.20, 0.25][:]:  # 11
                            for real_i in range(realizations)[:]:
                                fout = out_folder + "{}-attackedges-{}-T-{}-testratio-{}-swap_ratio-{}-gamma-{}-realization-{}-data-X.pkl".format(
                                    dataset, attack_edge, T, test_ratio, swap_ratio, gamma, real_i)
                                if not os.path.exists(fout):
                                    tasks.put(Task_generate_rn(V, E, Obs, T, test_ratio, swap_ratio, gamma, fout))
                                    num_jobs += 1
                    print "\n\nTttack_edge size {} Done.....................\n\n".format(attack_edge)

        for i in range(num_consumers):
            tasks.put(None)

        while num_jobs:
            results.get()
            num_jobs -= 1
            print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> remain: ",num_jobs

def pgd_sybils_data_generator():
    data_root = "/network/rit/lab/ceashpc/adil/data/adv_csl/Jan2/random_pgd/"
    # data_root = "./"
    realizations = 1
    attack_edges = [1000,2000,3000,4000,5000]
    network_files=["../data/Undirected_Facebook/facebook_sybils_attackedge",
                   "../data/enron/enron_attackedge",
                   "../data/slashdot/slashdot_sybils_attackedge"]
    tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()
    num_consumers = 50  # We only use 5 cores.
    # print 'Creating %d consumers' % num_consumers
    consumers = [Consumer(tasks, results)
                 for i in range(num_consumers)]
    for w in consumers:
        w.start()
    num_jobs=0
    for i,dataset in enumerate(["facebook","enron","slashdot"][:]):
        for attack_edge in [1000, 5000,10000,15000,20000][:]:
            filename = network_files[i]+"{}.pkl".format(attack_edge)
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
                        Obs = graph_process(V, E, T, swap_ratio)
                        for gamma in [0.0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.20, 0.25][:]:  # 11
                            for real_i in range(realizations)[:]:
                                fout = out_folder + "{}-attackedges-{}-T-{}-testratio-{}-swap_ratio-{}-gamma-{}-realization-{}-data-X.pkl".format(
                                    dataset, attack_edge, T, test_ratio, swap_ratio, gamma, real_i)
                                if not os.path.exists(fout):
                                    tasks.put(Task_generate_pgd(V, E, Obs, T, test_ratio, swap_ratio, gamma, fout))
                                    num_jobs += 1
                    print "\n\nTttack_edge size {} Done.....................\n\n".format(attack_edge)

        for i in range(num_consumers):
            tasks.put(None)

        while num_jobs:
            results.get()
            num_jobs -= 1
            print ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> remain: ",num_jobs


if __name__=='__main__':
    # analyze_data_FB()
    # analyze_data()

    random_flip_sybils_data_generator()
    random_noise_sybils_data_generator()
    # enron_simulation_data_generator()
    # slashdot_simulation_data_generator()
