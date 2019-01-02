__author__ = 'Feng Chen'
from multi_core_trust_inference import inference_apdm_format, list_sum, list_dot_scalar, calculate_measures
from log import Log
import os,sys
import time
import pickle
import numpy as np
import json,itertools,networkx as nx
#from SL_inference import *
#from SL_prediction import findPaths, SL_prediction
from basic_funs import *
from network_funs import *
import baseline
from feng_SL_inference import *
from multi_core_csl_inference_conflicting_evidence_new import inference_apdm_format as inference_apdm_format_conflict_evidence
#from SL_inference import *
from random import shuffle
import multiprocessing





"""
randomly select test_ratio of the edges in E e as testing edges, each of which has at least one one down or up edg
INPUT:
test_ratio:
V:
E:

OUTPUT
E_X: a subset of edges in V as testing edges
"""
def sample_X(test_ratio, V, E):
    n = len(V)
    n_E = len(E)
    rands = np.random.permutation(n_E)[:int(np.round(test_ratio * n_E))]
    edge_up_nns, edge_down_nns = get_edge_nns(V, E)
    E_X = [E[i] for i in rands if edge_up_nns.has_key(E[i]) or edge_down_nns.has_key(E[i])]
    return E_X


def post_process_data():
    methods = ["csl", "sl", "base1", "base2", "base3"][:]
    graph_sizes = [500, 1000, 5000, 10000, 47676]
    # graph_sizes = [2500, 7500]
    ratios = [0.2]
    realizations = 10
    for T in [6, 8, 11]:
        for swap_ratio in [0.00, 0.01, 0.05][:3]:                     #noise
            for test_ratio in [0.1, 0.2, 0.3, 0.4, 0.5][:]:           #percentage of edges to test (|E_x|/|E|)
                for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6][:]: #the percentage of edges set the observations to 1
                    for graph_size in graph_sizes[:3]:
                        for real_i in range(realizations)[:]:
                            print "real_i: {}, ratio: {}, test_ratio: {}, swaprate: {}, graph_size: {}".format(real_i, ratio, test_ratio, swap_ratio, graph_size)
                            f = "/home/apdm02/workspace/git/data/cls_conflict/trust-analysis3/nodes-{}-T-{}-rate-{}-swaprate-{}-realization-{}-data.pkl".format(graph_size, T, ratio, swap_ratio, real_i)
                            fout = "/home/apdm02/workspace/git/data/cls_conflict/trust-analysis3/nodes-{}-T-{}-rate-{}-testratio-{}-swaprate-{}-realization-{}-data-X.pkl".format(graph_size, T, ratio, test_ratio, swap_ratio, real_i)
                            if not os.path.exists(fout) or True:
                                print f
                                pkl_file = open(f, 'rb')
                                [V, E, Obs] = pickle.load(pkl_file)
                                n = len(V)
                                n_E = len(E)
                                _, edge_down_nns, id_2_edge, edge_2_id, _, _ = reformat(V, E, Obs)
                                _, dict_paths = generate_eopinion_PSL_rules_from_edge_cnns(edge_down_nns, id_2_edge, edge_2_id)
                                E_X = sample_X(test_ratio, V, [id_2_edge[item] for item in dict_paths.keys()])
                                pkl_file = open(fout, 'wb')
                                pickle.dump([V, E, Obs, E_X], pkl_file)
                                pkl_file.close()


def post_process_data_conflict_evidence():
    methods = ["csl", "sl", "base1", "base2", "base3"][:]
    graph_sizes = [500, 1000, 5000, 10000, 47676]
    # graph_sizes = [2500, 7500]
    ratios = [0.2]
    realizations = 10
    for T in [6, 8, 11][:]:
        for swap_ratio in [0.00, 0.01, 0.05][:]:                         #noise
            for test_ratio in [0.1, 0.2, 0.3, 0.4, 0.5][:]:              #percentage of edges to test (|E_x|/|E|)
                for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6][:]:    #the percentage of edges set the observations to 1
                    for ratio_conflict in [0.1, 0.2, 0.3, 0.4, 0.5,0.6]:
                        for graph_size in graph_sizes[:3]:
                            for real_i in range(realizations)[:]:
                                print "real_i: {}, ratio: {}, test_ratio: {}, swaprate: {}, graph_size: {}".format(real_i, ratio, test_ratio, swap_ratio, graph_size)
                                f = "/home/apdm02/workspace/git/data/cls_conflict/trust-analysis3/nodes-{}-T-{}-rate-{}-testratio-{}-swaprate-{}-realization-{}-data-X.pkl".format(graph_size, T, ratio, test_ratio, swap_ratio, real_i)
                                fout = "/home/apdm02/workspace/git/data/cls_conflict/trust-analysis3/nodes-{}-T-{}-rate-{}-testratio-{}-swaprate-{}-confictratio-{}-realization-{}-data-X.pkl".format(graph_size, T, ratio, test_ratio, swap_ratio, ratio_conflict, real_i)
                                if not os.path.exists(fout) or True:
                                    print f
                                    pkl_file = open(f, 'rb')
                                    [V, E, Obs, E_X] = pickle.load(pkl_file)
                                    n = len(V)
                                    n_E = len(E)
                                    E_Y = [e for e in E if e not in E_X]
                                    rand_seq_E_Y = copy.deepcopy(E_Y)
                                    shuffle(rand_seq_E_Y)
                                    cnt = int(np.round(len(E_Y) * ratio_conflict))
                                    X_b = rand_seq_E_Y[:cnt]
                                    T = len(Obs[E[0]])
                                    mid = int(np.round(T/2.0))
                                    for e in X_b:
                                        for t in range(mid,T):
                                            Obs[e][t] = 1 - Obs[e][t]
                                    pkl_file = open(fout, 'wb')
                                    pickle.dump([V, E, Obs, E_X, X_b], pkl_file)
                                    pkl_file.close()


def main():
    # post_process_data()
    # post_process_data_conflict_evidence()
    # experiment_proc()
    experiment_proc_server_SL_traffic()
    # experiment_proc_server_SL()



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
class Task_evaluate(object):
    def __init__(self, V, E, Obs, Omega, E_X, X_b,logging, method,graph_size,T,swap_ratio,test_ratio,ratio,ratio_conflict,real_i):
        self.V = V
        self.E = E
        self.Obs =Obs
        self.Omega = Omega
        self.E_X = E_X
        self.X_b = X_b
        self.logging = logging
        self.method = method
        self.graph_size = graph_size
        self.T = T
        self.swap_ratio = swap_ratio
        self.test_ratio = test_ratio
        self.ratio = ratio
        self.ratio_conflict = ratio_conflict
        self.real_i = real_i

    def __call__(self):
        running_starttime = time.time()
        pred_omega_x = SL_prediction(self.V, self.E, self.Obs, self.Omega, copy.deepcopy(self.E_X))
        alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested = calculate_measures(
            self.Omega, pred_omega_x, self.E_X, self.logging)
        running_endtime = time.time()
        running_time = running_endtime - running_starttime
        # print "accuracy: {}, prob_mse: {}".format(accuracy, prob_mse)
        return accuracy, alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, running_time,self.method,self.graph_size,self.T,self.swap_ratio,self.test_ratio,self.ratio,self.ratio_conflict,self.real_i


    def __str__(self):
        return '%s' % (self.p0)
class Task_evaluate_traffic(object):
    def __init__(self, V, E, Obs, Omega, E_X, X_b, logging, method,dataset, weekday, hour, ref_ratio, test_ratio, ratio_conflict, real_i):
        self.V = V
        self.E = E
        self.Obs =Obs
        self.Omega = Omega
        self.E_X = E_X
        self.X_b = X_b
        self.logging = logging
        self.method = method
        self.dataset = dataset
        self.weekday = weekday
        self.hour = hour
        self.test_ratio = test_ratio
        self.ref_ratio = ref_ratio
        self.ratio_conflict = ratio_conflict
        self.real_i = real_i

    def __call__(self):
        running_starttime = time.time()
        pred_omega_x = SL_prediction(self.V, self.E, self.Obs, self.Omega, copy.deepcopy(self.E_X))
        alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested = calculate_measures(
            self.Omega, pred_omega_x, self.E_X, self.logging)
        running_endtime = time.time()
        running_time = running_endtime - running_starttime
        # print "accuracy: {}, prob_mse: {}".format(accuracy, prob_mse)
        return accuracy, alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, running_time,self.method,self.dataset, self.weekday, self.hour, self.ref_ratio, self.test_ratio, self.ratio_conflict, self.real_i


    def __str__(self):
        return '%s' % (self.p0)



def experiment_proc_server_SL_traffic():
    logging = Log()
    data_root = "/network/rit/lab/ceashpc/adil/data/csl-data/June25/"
    report_stat = False
    ref_ratios = [0.9, 0.8, 0.6]
    realizations = 10
    datasets = ['philly', 'dc']
    count = 0
    T = 43
    methods = ["sl", "csl", "csl-conflict-1", "csl-conflict-2", "base1", "base2", "base3"][:1]
    for dataset in datasets[:]:
        dataroot = data_root + dataset + "/"
        for weekday in range(5)[:1]:
            for hour in range(7, 22)[1:2]:
                for ref_ratio in ref_ratios[:2]:
                    tasks = multiprocessing.Queue()
                    results = multiprocessing.Queue()
                    num_consumers = 50  # We only use 5 cores.
                    print 'Creating %d consumers' % num_consumers
                    consumers = [Consumer(tasks, results)
                                 for i in range(num_consumers)]
                    for w in consumers:
                        w.start()
                    num_jobs = 0
                    for test_ratio in [0.1, 0.2, 0.3, 0.4, 0.5][:]:
                        for ratio_conflict in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6][:]:
                            for real_i in range(realizations)[:]:
                                logging.write(
                                    str(count) + " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`")
                                count += 1.0
                                for method in methods[:]:
                                    f = dataroot + '/network_{}_weekday_{}_hour_{}_refspeed_{}-testratio-{}-confictratio-{}-realization-{}.pkl'.format(
                                        dataset, weekday, hour, ref_ratio, test_ratio, ratio_conflict, real_i)
                                    logging.write(
                                        "dataset: {} method: {}, weekday:{},hour:{},T, {},ref_ratio:{}, test_ratio: {},conflict_ratio:{}".format(
                                            dataset, method, weekday, hour, T, ref_ratio, test_ratio, ratio_conflict))
                                    logging.write(f)
                                    pkl_file = open(f, 'rb')
                                    [V, E, Obs, E_X, X_b] = pickle.load(pkl_file)
                                    m_idx = int(round(len(Obs[E[0]]) / 2.0))
                                    t_Obs = {e: e_Obs[m_idx - 5:m_idx + 6] for e, e_Obs in Obs.items()}
                                    Omega = calc_Omega_from_Obs2(t_Obs, E)

                                    tasks.put(Task_evaluate_traffic(V, E, t_Obs, Omega, E_X, X_b, logging, method,dataset, weekday, hour, ref_ratio, test_ratio, ratio_conflict, real_i))
                                    num_jobs+=1

                    # Add a poison pill for each consumer
                    for i in xrange(num_consumers):
                        tasks.put(None)

                    while num_jobs:
                        accuracy, alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, running_time, method, dataset, weekday, hour, ref_ratio, test_ratio, ratio_conflict, real_i = results.get()
                        result_ = {'dataset': dataset, 'weekday': weekday, 'hour': hour, 'ref_ratio': ref_ratio,
                                   'sample_size': 1, 'T': T, 'ratio_conflict': ratio_conflict,
                                   'test_ratio': test_ratio, 'acc': (accuracy, 0.0),
                                   'prob_mse': (prob_mse, 0.0),'alpha_mse': (alpha_mse, 0), 'beta_mse': (beta_mse, 0),
                                   'u_mse': (u_mse, 0), 'b_mse': (b_mse, 0), 'd_mse': (d_mse, 0), 'realization': real_i, 'runtime': running_time}
                        outf = '../output/test/{}_results-server-traffic-June25-T11.json'.format(method)
                        with open(outf, 'a') as outfp:
                            outfp.write(json.dumps(result_) + '\n')
                            sys.stdout.write(str(num_jobs)+" ")
                            if num_jobs%100==0:
                                sys.stdout.write("\n")
                        num_jobs -= 1
                    print "\n\n weekday:",weekday,"is Done......\n\n"

def experiment_proc_server_SL():
    logging = Log()
    # data_root = "/network/rit/lab/ceashpc/adil/data/csl-data/apr21/"

    data_root = "/network/rit/lab/ceashpc/adil/data/csl-data/May23/"
    methods = ["csl", "csl-3-rules", "csl-3-rules-conflict-evidence", "sl", "base1", "base2", "base3"][3:4]
    # graph_sizes = [500, 1000, 5000, 10000, 47676]
    graph_sizes = [8518,1000, 5000,10000,47676]
    ratios = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    realizations = 10
    for graph_size in graph_sizes[1:2]:
        for ratio in [0.0,0.6, 0.7, 0.8][1:]:
            tasks = multiprocessing.Queue()
            results = multiprocessing.Queue()
            num_consumers = 50    # We only use 5 cores.
            print 'Creating %d consumers' % num_consumers
            consumers = [Consumer(tasks, results)
                         for i in range(num_consumers)]
            for w in consumers:
                w.start()
            num_jobs = 0
            for T in [8,9,10,11][:]:
                for swap_ratio in [0.00, 0.01, 0.05][:1]:
                    for test_ratio in [0.1, 0.2, 0.3, 0.4, 0.5][:]:                #percentage of edges to test (|E_x|/|E|)
                             #the percentage of edges set the observations to 1
                        for ratio_conflict in [0.0,0.1, 0.2, 0.3, 0.4, 0.5,0.6][:]:
                            for real_i in range(realizations)[:]:
                                data_folder = data_root + str(graph_size) + "/"
                                # out_folder = data_root + "/"
                                logging.write(str(num_jobs)+" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                                for method in methods[:]:
                                    f = data_folder + "nodes-{}-T-{}-rate-{}-testratio-{}-swaprate-{}-confictratio-{}-realization-{}-data-X.pkl".format(
                                        graph_size, T, ratio, test_ratio, swap_ratio, ratio_conflict, real_i)

                                    logging.write("method: {}, T, {}, real_i: {}, ratio: {}, test_ratio: {}, swaprate: {}, graph_size: {}".format(method, T, real_i, ratio, test_ratio, swap_ratio, graph_size))
                                    logging.write(f)
                                    pkl_file = open(f, 'rb')
                                    [V, E, Obs, E_X, X_b] = pickle.load(pkl_file)
                                    pkl_file.close()
                                    Omega = calc_Omega_from_Obs2(Obs, E)
                                    E_X={e:1 for e in E_X}
                                    tasks.put(Task_evaluate(V, E, Obs, Omega, E_X, X_b, logging, method,graph_size,T,swap_ratio,test_ratio,ratio,ratio_conflict,real_i))
                                    num_jobs+=1

            # Add a poison pill for each consumer
            for i in xrange(num_consumers):
                tasks.put(None)

            while num_jobs:
                accuracy, alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, running_time, method, graph_size, T, swap_ratio, test_ratio, ratio, ratio_conflict, real_i = results.get()
                result_ = {'network_size': graph_size, 'positive_ratio': ratio, "realization": real_i,
                           'sample_size': 1, 'T': T, 'ratio_conflict': ratio_conflict,
                           'test_ratio': test_ratio, 'acc': (accuracy, 0),
                           'alpha_mse': (alpha_mse, 0), 'beta_mse': (beta_mse, 0),
                           'u_mse': (u_mse, 0), 'b_mse': (b_mse, 0), 'd_mse': (d_mse, 0),
                           'prob_mse': (prob_mse, 0), 'runtime': running_time}
                outf = '../output/test/{}_results-server-June5-rnd2-18.json'.format(method)
                with open(outf, 'a') as outfp:
                    outfp.write(json.dumps(result_) + '\n')
                    sys.stdout.write(str(num_jobs)+" ")
                    if num_jobs%100==0:
                        sys.stdout.write("\n")
                num_jobs -= 1
            print "\n\npos ratio:",ratio,"is Done......\n\n"





"""
INPUT
V = [0, 1, ...] is a list of vertex ids
E: a list of pairs of vertex ids
Obs: a dictionary with key edge and its value a list of congestion observations of this edge from t = 1 to t = T
Omega: a dictionary with key edge and its value a pair of alpha and beta.
E_X: Set of edges (pairs of nodes)
logging: log object for logging intermediate information
method: 'csl', 'sl', 'base1', 'base2', 'base3'
init_alpha_beta: the initial pair of alpha and beta for the edges in E_X
psl: True if we want to use pure PSL inference, instead of the inference based on the proposed model; False, otherwise.
approx: True if we want to try the probability values {0.01, 0.02, ..., 0.99} to solve each cubic function equation; False and calculate the exact solution, otherwise.

OUTPUT
It will print out the following information:

1. The true uncertainty mass and the predicted uncertainty mass
1. The true expected probability and predicted expected probability
2. The MSE on uncertainty mass
4. The MSE on expected probability
""""csl-3-rules", "csl-3-rules-conflict-evidence"
def evaluate(V, E, Obs, Omega, E_X, X_b, logging, method = 'csl', psl = False, approx = False, init_alpha_beta = (1, 1), report_stat = False):
    running_starttime = time.time()
    if method == 'sl':
        pred_omega_x = SL_prediction(V, E, Obs, Omega, copy.deepcopy(E_X))
        # print pred_omega_x
    elif method == 'csl':
        pred_omega_x = inference_apdm_format(V, E, Obs, Omega, E_X, logging)
    elif method == 'csl-3-rules':
        print method
        # b = {e: 0 for e in E}
        b={}
        # X_b = []
        t1=time.time()
        X_b = {e: 0 for e in E if not E_X.has_key(e)}
        psl = True
        print "X_b cons",time.time()-t1
        pred_omega_x, _ = inference_apdm_format_conflict_evidence(V, E, Obs, Omega, b, X_b, E_X, logging, psl)
    elif method == 'csl-3-rules-conflict-evidence':
        # b = {e: 0 for e in E}
        b={}
        # X_b = {e: 0 for e in E if e not in E_X}
        X_b = {e: 0 for e in E if e not in E_X}
        psl = False
        pred_omega_x, _ = inference_apdm_format_conflict_evidence(V, E, Obs, Omega, b, X_b, E_X, logging, psl)
    elif method == 'base1':
        pred_omega_x = baseline.base1(V, E, Obs, Omega, E_X)
    elif method == 'base2':
        pred_omega_x = baseline.base2(V, E, Obs, Omega, E_X)
    elif method == 'base3':
        pred_omega_x = baseline.base3(V, E, Obs, Omega, E_X)
    else:
        raise Exception("Method Error")
    # print "pred_omega_x", pred_omega_x
    alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested = calculate_measures(Omega, pred_omega_x, E_X, logging)
    running_endtime = time.time()
    running_time = running_endtime - running_starttime
    #print "accuracy: {}, prob_mse: {}".format(accuracy, prob_mse)
    return accuracy, alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, running_time



if __name__=='__main__':
    # generateData()

    main()

