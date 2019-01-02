__author__ = 'Feng Chen'
from multi_core_trust_inference import inference_apdm_format, list_sum, list_dot_scalar, calculate_measures
from log import Log
import os
import time
import pickle
import numpy as np
import json,itertools
#from SL_inference import *
#from SL_prediction import findPaths, SL_prediction
from basic_funs import *
from network_funs import *
import baseline
from feng_SL_inference_multiCore import *

from multi_core_csl_inference_conflicting_evidence_new_numpy import inference_apdm_format as inference_apdm_format_conflict_evidence

#from SL_inference import *
from random import shuffle

def generateData(percent=0.05):
    dw = DataWrapper()
    V, E, Obs, G = dw.data_wrapper()

    Omega = obs_to_omega(Obs)
    sizeE = len(E)
    E_X = random.sample(E,int(round(sizeE*percent)))
    if len(E_X) == 0:
        raise Exception('HAHA EMPTY X HERE')

    print('number of nodes: {}'.format(len(V)))
    print('number of edges: {}'.format(len(E)))
    print('number of cc: {}'.format(nx.number_connected_components(G)))
    outfile = open('data/data.pkl','w')

    pickle.dump([V,E,Omega,Obs,E_X],outfile)



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


def calc_Omega_from_Obs3(Obs, E,E_X):
    W=2.0
    a=0.5
    T = len(Obs.values()[0])
    Omega = {}
    Opinions={}
    dem_=[]
    for e in E_X:
        pos_evidence=np.sum(Obs[e])* 1.0
        neg_evidence=T - pos_evidence
        # print e,pos_evidence,neg_evidence
        # dem_.append(pos_evidence + neg_evidence + W)
        b = pos_evidence/(pos_evidence+neg_evidence+W)
        d = neg_evidence / (pos_evidence + neg_evidence + W)
        u = W/(pos_evidence + neg_evidence + W)
        alpha=W*b/u + W*a
        beta= W*d/u + W*(1-a)
        Opinions[e]=(b,d,u)
        Omega[e] = (alpha,beta)
    # print set(dem_)
    return Opinions

def Omega_2_opinion(pred_omega,E_X):
    W=2.0
    a=0.5
    # T = len(Obs.values()[0])
    Omega = {}
    Opinions={}
    # dem_=[]
    for e in E_X.keys():
        r=np.abs(pred_omega[e][0]-W*a)
        s=np.abs(pred_omega[e][1]-W*(1.0-a))
        # print ">",e,r,s
        # dem_.append(r+s+W)
        b=r/(r+s+W)
        d=s/(r+s+W)
        u=W/(r+s+W)
        # b = (pred_omega[0]-1.0)/(pred_omega[0]+pred_omega[1])
        # d = (pred_omega[1]-1.0)/(pred_omega[0]+pred_omega[1])
        # u = W/(pred_omega[0]+pred_omega[1])
        Opinions[e]=(b,d,u)
    # print set(dem_)
    return Opinions

def main():
    # post_process_data()
    # post_process_data_conflict_evidence()
    # experiment_proc()
    experiment_proc_server()
    # experiment_proc_server_checkHist()


def experiment_proc_server():
    logging = Log()
    data_root = "/network/rit/lab/ceashpc/adil/data/csl-data/Oct10/"  #May23
    methods = ["csl", "csl-3-rules", "csl-3-rules-conflict-evidence", "sl", "base1", "base2", "base3"][:4]
    # graph_sizes = [500, 1000, 5000, 10000, 47676]
    graph_sizes = [8518,1000, 5000,10000,47676]
    ratios = [0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8]
    realizations = 1
    case_count=0
    result = {}
    for graph_size in graph_sizes[2:3]:
        for T in [8, 9, 10, 11,21][4:]:    #5,6,10,20,11,21,15
            for real_i in range(realizations)[:1]:
                for ratio in [0.2,0.3,0.6,0.7,0.8][:1]:#0.0,0.1,0.2,0.3,the percentage of edges set the observations to 1
                    for swap_ratio in [0.00, 0.01, 0.05][:1]:

                        for test_ratio in [0.1, 0.2, 0.3, 0.4, 0.5][:]:                #percentage of edges to test (|E_x|/|E|)
                            for ratio_conflict in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,0.6][1:5]:
                                out_folder = data_root + str(graph_size) + "/"
                                logging.write(str(case_count)+" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                                case_count += 1.0

                                outf = '../output/test/results-server-sep27-{}-0.2-tr-ConfRatio-val-t21-T4.pkl'.format(graph_size)
                                for method in methods[:]:
                                    f = out_folder + "nodes-{}-T-{}-rate-{}-testratio-{}-swaprate-{}-confictratio-{}-realization-{}-data-X.pkl".format(
                                        graph_size, T, ratio, test_ratio, swap_ratio, ratio_conflict, real_i)
                                    logging.write("method: {}, T, {}, real_i: {}, ratio: {}, test_ratio: {}, swaprate: {}, graph_size: {}".format(method, T, real_i, ratio, test_ratio, swap_ratio, graph_size))
                                    logging.write(f)
                                    pkl_file = open(f, 'rb')
                                    [V, E, Obs, E_X, X_b] = pickle.load(pkl_file)

                                    E_X = {e: 1 for e in E_X}
                                    # print test_ratio,len(E_X)

                                    Omega = calc_Omega_from_Obs2(Obs, E)
                                    True_Opinions=calc_Omega_from_Obs3(Obs,E,E_X)
                                    pre_Opinions,accuracy, alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, running_time = evaluate(V, E, Obs, Omega, E_X, X_b, logging, method)
                                    logging.write("prob_mse: {}, running time: {}".format(prob_mse, running_time))
                                    result[method+"-"+str(test_ratio)+"-"+str(ratio_conflict)+"-"+str(T)] = (True_Opinions,pre_Opinions)
                                    # print result_

            print "\n\nT=",T,"is done.......................................\n\n"
        outfp = open(outf, 'a')
        pickle.dump(result, outfp)
        outfp.close()
        print "\n\nN=", graph_size, "is done.........................................\n\n"



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
        pred_omega_x = SL_prediction_multiCore(V, E, Obs, Omega, copy.deepcopy(E_X))
        pre_opinions = Omega_2_opinion(pred_omega_x, E_X)
        # print pred_omega_x
    elif method == 'csl':
        pred_omega_x = inference_apdm_format(V, E, Obs, Omega, E_X, logging)
        pre_opinions = Omega_2_opinion(pred_omega_x,E_X)
    elif method == 'csl-3-rules':
        # b = {e: 0 for e in E}
        b={}
        # X_b = []
        X_b = {e: 0 for e in E if not E_X.has_key(e)}
        psl = True
        pred_omega_x, _ = inference_apdm_format_conflict_evidence(V, E, Obs, Omega, b, X_b, E_X, logging, psl)
        pre_opinions = Omega_2_opinion(pred_omega_x, E_X)
    elif method == 'csl-3-rules-conflict-evidence':
        # b = {e: 0 for e in E}
        b={}
        # X_b = {e: 0 for e in E if e not in E_X}
        X_b = {e: 0 for e in E if not E_X.has_key(e)}
        psl = False
        pred_omega_x, _ = inference_apdm_format_conflict_evidence(V, E, Obs, Omega, b, X_b, E_X, logging, psl)
        pre_opinions = Omega_2_opinion(pred_omega_x, E_X)
    elif method == 'base1':
        pred_omega_x = baseline.base1(V, E, Obs, Omega, E_X)
    elif method == 'base2':
        pred_omega_x = baseline.base2(V, E, Obs, Omega, E_X)
    elif method == 'base3':
        pred_omega_x = baseline.base3(V, E, Obs, Omega, E_X)
    else:
        raise Exception("Method Error")
    # print "pred_omega_x", len(pred_omega_x),len(E_X)
    alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested = calculate_measures(Omega, pred_omega_x, E_X, logging)
    running_endtime = time.time()
    running_time = running_endtime - running_starttime
    #print "accuracy: {}, prob_mse: {}".format(accuracy, prob_mse)
    return pre_opinions,accuracy, alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, running_time



if __name__=='__main__':
    # generateData()

    main()

