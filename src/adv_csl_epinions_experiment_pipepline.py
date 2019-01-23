__author__ = 'Adil Alim'
from multi_core_trust_inference_csl import inference_apdm_format, list_sum, list_dot_scalar, calculate_measures
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
# import baseline
from SL_inference_multiCore import *

from multi_core_csl_inference_adversarial_epinions import inference_apdm_format as inference_apdm_format_conflict_evidence
from multi_core_csl_inference_no_adversarial_epinions import inference_apdm_format as inference_NoAdvTraining

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



def main():
    # post_process_data()
    # post_process_data_conflict_evidence()
    # experiment_proc()
    experiment_proc_server()
    # experiment_proc_server_checkHist()


def experiment_proc_server():
    logging = Log()
    data_root = "/network/rit/lab/ceashpc/adil/data/adv_csl/Jan2/"  #May23 May23-3

    methods = ["SL","CSL", "Adv-CSL"][2:]
    # graph_sizes = [500, 1000, 5000, 10000, 47676]
    graph_sizes = [1000, 5000,10000,47676]
    ratios = [0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8]
    realizations = 10
    case_count=0
    for real_i in range(realizations)[1:2]:
        for adv_type in ["random_noise","random_pgd","random_pgd_csl","random_pgd_gcn_vae"][:]:
            for graph_size in graph_sizes[1:2]:
                for T in [8, 9, 10, 11][2:3]:    #5,6,10,20,11,21,15
                    for ratio in [0.2][:]:#0.0,0.1,0.2,0.3,the percentage of edges set the observations to 1
                        for swap_ratio in [0.00, 0.01, 0.05][:1]:
                            for test_ratio in [0.3,0.1, 0.2, 0.4, 0.5][:1]:                #percentage of edges to test (|E_x|/|E|)
                                for gamma in [0.0, 0.01, 0.03, 0.05, 0.07,0.09,0.2,0.3,0.4,0.5][:]:  # 8
                                    out_folder = data_root +"/"+adv_type +"/"+ str(graph_size) + "/"
                                    logging.write(str(case_count)+" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                                    case_count += 1.0
                                    for method in methods[:]:
                                        f = out_folder + "nodes-{}-T-{}-rate-{}-testratio-{}-swaprate-{}-gamma-{}-realization-{}-data-X.pkl".format(
                                            graph_size, T, ratio, test_ratio, swap_ratio, gamma, real_i)
                                        outf = '../output/epinions/{}_results-server-{}-Jan22-debug-{}.json'.format(method,graph_size,adv_type)

                                        logging.write("method: {}, T, {}, real_i: {}, ratio: {}, test_ratio: {}, swaprate: {},gamma:{}, graph_size: {}".format(method, T, real_i, ratio, test_ratio, swap_ratio,gamma, graph_size))
                                        logging.write(f)
                                        pkl_file = open(f, 'rb')
                                        [V, E, Obs, E_X, X_b] = pickle.load(pkl_file)
                                        n = len(V)
                                        n_E = len(E)
                                        ndays = len(Obs[E[0]])
                                        E_X = {e: 1 for e in E_X}
                                        accuracys = []
                                        prob_mses = []
                                        u_mses = []
                                        b_mses = []
                                        d_mses = []
                                        alpha_mses = []
                                        beta_mses = []
                                        running_times = []
                                        nposi = 0
                                        nnega = 0
                                        for start_t in range(ndays - T + 1)[:1]:
                                            t_Obs = {e: e_Obs[start_t:start_t+T] for e, e_Obs in Obs.items()}
                                            Omega = calc_Omega_from_Obs2(t_Obs, E)
                                            accuracy, alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, running_time = evaluate(V, E, t_Obs, Omega, E_X, X_b, logging, method)
                                            i_nposi, i_nnega = accuracy_2_posi_nega(accuracy)
                                            nposi += i_nposi
                                            nnega += i_nnega
                                            b_mses.append(b_mse)
                                            d_mses.append(d_mse)
                                            alpha_mses.append(alpha_mse)
                                            beta_mses.append(beta_mse)
                                            accuracys.append(accuracy)
                                            prob_mses.append(prob_mse)
                                            u_mses.append(u_mse)
                                            running_times.append(running_time)
                                            # print accuracy, prob_mse, running_time
                                        mu_alpha_mse = np.mean(alpha_mses)
                                        sigma_alpha_mse = np.std(alpha_mses)
                                        mu_beta_mse = np.mean(beta_mses)
                                        sigma_beta_mse = np.std(beta_mses)
                                        mu_u_mse = np.mean(u_mses)
                                        sigma_u_mse = np.std(u_mses)
                                        mu_b_mse = np.mean(b_mses)
                                        sigma_b_mse = np.mean(b_mses)
                                        mu_d_mse = np.mean(d_mses)
                                        sigma_d_mse = np.mean(d_mses)
                                        mu_accuracy = np.mean(accuracys)
                                        sigma_accuracy = np.std(accuracys)
                                        mu_prob_mse = np.mean(prob_mses)
                                        sigma_prob_mse = np.std(prob_mses)
                                        running_time = np.mean(running_times)
                                        logging.write("prob_mse: {}, running time: {} gamma:{} TR:{}".format(mu_prob_mse, running_time,gamma,test_ratio))

                                        # logging.write(info)
                                        result_ = {'network_size': graph_size,'adv_type':adv_type, 'positive_ratio': ratio, "realization": real_i, 'sample_size': ndays - T + 1, 'T': T,'gamma':gamma,
                                                'test_ratio': test_ratio, 'acc': (mu_accuracy, sigma_accuracy), 'alpha_mse': (mu_alpha_mse, sigma_alpha_mse), 'beta_mse': (mu_beta_mse, sigma_beta_mse), 'u_mse': (mu_u_mse, sigma_u_mse), 'b_mse': (mu_b_mse, sigma_b_mse), 'd_mse': (mu_d_mse, sigma_d_mse), 'prob_mse': (mu_prob_mse, sigma_prob_mse), 'runtime': running_time}
                                        # print result_
                                        outfp = open(outf, 'a')
                                        outfp.write(json.dumps(result_) + '\n')
                                        outfp.close()
                print "\n\nT=",T,"is done.......................................\n\n"
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
    if method == 'SL':
        pred_omega_x = SL_prediction_multiCore(V, E, Obs, Omega, copy.deepcopy(E_X))
        # print pred_omega_x
    elif method == 'CSL':
        pred_omega_x = inference_apdm_format(V, E, Obs, Omega, E_X, logging)
    elif method == 'NAT-CSL':
        # b = {e: 0 for e in E}
        b = {}
        # X_b = {e: 0 for e in E if e not in E_X}
        X_b = {e: 0 for e in E if not E_X.has_key(e)}
        psl = False
        pred_omega_x, _ = inference_NoAdvTraining(V, E, Obs, Omega, b, X_b, E_X, logging, psl)
    elif method == 'Adv-CSL':
        # b = {e: 0 for e in E}
        b={}
        # X_b = {e: 0 for e in E if e not in E_X}
        X_b = {e: 0 for e in E if not E_X.has_key(e)}
        psl = False
        pred_omega_x, _ = inference_apdm_format_conflict_evidence(V, E, Obs, Omega, b, X_b, E_X, logging, psl)
    else:
        raise Exception("Method Error")
    # print "pred_omega_x", len(pred_omega_x),len(E_X)
    alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested = calculate_measures(Omega, pred_omega_x, E_X, logging)
    running_endtime = time.time()
    running_time = running_endtime - running_starttime
    #print "accuracy: {}, prob_mse: {}".format(accuracy, prob_mse)
    return accuracy, alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, running_time



if __name__=='__main__':
    # generateData()

    main()

