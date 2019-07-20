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
from SL_inference import *
from multi_core_csl_inference_conflicting_evidence_new import inference_apdm_format as inference_apdm_format_conflict_evidence
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


"""
0-> 1 -> 2
^   ^    ^
3-> 4 -> 5
^   ^    ^
6-> 7 -> 8
"""
def testcase5():
    logging = Log('Testcase5.txt')
    V = [0,1,2,3,4,5,6,7,8]
    E = [(0,1), (1,2), (3,4), (4,5), (6,7), (7,8), (3,0), (6,3), (4,1), (7,4), (5,2), (8,5), (7,5), (7,2), (4,2), (6,4), (3,1), (6,1)]
    Obs = {(0,1): [0], (1,2): [0], (3,4): [0], (4,5): [0], (6,7): [0], (7,8): [0], (3,0): [0], (6,3): [0], (4,1): [0], (7,4): [0], (5,2): [0], (8,5): [0], (7,5): [0], (7,2): [0], (4,2): [0], (6,4): [0], (3,1): [0], (6,1): [0]}
    # for e in Obs.keys():
    #     Obs[e] = [1]


    """
    0-> 1 -> 2
    ^   ^    ^
    3-> 4 -> 5
    ^   ^    ^
    6-> 7 -> 8
    """
    Obs[(7,8)] = [1]
    Obs[(8,5)] = [1]
    Obs[(7,5)] = [1]
    E_X = [(7,5), (4,2), (7,2), (3,1)]

    logging.write("\r\n\r\nExample 1: ##########################")
    logging.write("Only (7,5) should be congested")
    logging.write("init_alpha_beta = (1, 1), which means equal chance to be congested or uncongested!\r\n ")
    Omega = {}
    for e in E:
        if np.mean(Obs[e]) > 0.5:
            Omega[e] = (10,1)
        else:
            Omega[e] = (1, 10)
    method = "csl"
    evaluate(V, E, Obs, Omega, E_X, logging, method, psl = False, approx = False, init_alpha_beta = (1, 1), report_stat = True)


    # return

    """
    0-> 1 -> 2
    ^   ^    ^
    3-> 4 -> 5
    ^   ^    ^
    6-> 7 -> 8
    """
    Obs = {(0,1): [0], (1,2): [0], (3,4): [0], (4,5): [0], (6,7): [0], (7,8): [0], (3,0): [0], (6,3): [0], (4,1): [0], (7,4): [0], (5,2): [0], (8,5): [0], (7,5): [0], (7,2): [0], (4,2): [0], (6,4): [0], (3,1): [0], (6,1): [0]}
    Obs[(4,1)] = [1]
    Obs[(1,2)] = [1]
    Obs[(4,2)] = [1]
    E_X = [(7,5), (4,2), (7,2), (3,1)]

    logging.write("\r\n\r\nExample 2: ##########################")
    logging.write("Only (4,2) should be congested")
    logging.write("init_alpha_beta = (1, 1), which means equal chance to be congested or uncongested!\r\n ")
    Omega = {}
    for e in E:
        if np.mean(Obs[e]) > 0.5:
            Omega[e] = (10,1)
        else:
            Omega[e] = (1, 10)
    evaluate(V, E, Obs, Omega, E_X, logging, method, psl = False, approx = False, init_alpha_beta = (1, 1), report_stat = True)

    # return

    """
    0-> 1 -> 2
    ^   ^    ^
    3-> 4 -> 5
    ^   ^    ^
    6-> 7 -> 8
    """
    Obs = {(0,1): [0], (1,2): [0], (3,4): [0], (4,5): [0], (6,7): [0], (7,8): [0], (3,0): [0], (6,3): [0], (4,1): [0], (7,4): [0], (5,2): [0], (8,5): [0], (7,5): [0], (7,2): [0], (4,2): [0], (6,4): [0], (3,1): [0], (6,1): [0]}
    Obs[(4,1)] = [1]
    Obs[(0,1)] = [1]
    Obs[(4,2)] = [0]
    E_X = [(7,5), (4,2), (7,2), (3,1)]

    logging.write("\r\n\r\nExample 3: ##########################")
    logging.write("No edges should be congested")
    logging.write("init_alpha_beta = (1, 1), which means equal chance to be congested or uncongested!\r\n ")
    Omega = {}
    for e in E:
        if np.mean(Obs[e]) > 0.5:
            Omega[e] = (10,1)
        else:
            Omega[e] = (1, 10)
    evaluate(V, E, Obs, Omega, E_X, logging, method, psl = False, approx = False, init_alpha_beta = (1, 1), report_stat = True)

    # return


    """
    0-> 1 -> 2
    ^   ^    ^
    3-> 4 -> 5
    ^   ^    ^
    6-> 7 -> 8
    """
    Obs = {(0,1): [0], (1,2): [0], (3,4): [0], (4,5): [0], (6,7): [0], (7,8): [0], (3,0): [0], (6,3): [0], (4,1): [0], (7,4): [0], (5,2): [0], (8,5): [0], (7,5): [0], (7,2): [0], (4,2): [0], (6,4): [0], (3,1): [0], (6,1): [0]}
    Obs[(7,8)] = [1]
    Obs[(8,5)] = [1]
    Obs[(5,2)] = [1]
    Obs[(7,5)] = [1]
    Obs[(7,2)] = [1]
    E_X = [(7,5), (4,2), (7,2), (3,1)]

    logging.write("\r\n\r\nExample 4: ##########################")
    logging.write("No edges should be congested")
    logging.write("init_alpha_beta = (1, 1), which means equal chance to be congested or uncongested!\r\n ")
    Omega = {}
    for e in E:
        if np.mean(Obs[e]) > 0.5:
            Omega[e] = (10,1)
        else:
            Omega[e] = (1, 10)
    evaluate(V, E, Obs, Omega, E_X, logging, method, psl = False, approx = False, init_alpha_beta = (1, 1), report_stat = True)

    # return


    """
    0-> 1 -> 2
    ^   ^    ^
    3-> 4 -> 5
    ^   ^    ^
    6-> 7 -> 8
    """
    Obs = {(0,1): [0], (1,2): [0], (3,4): [0], (4,5): [0], (6,7): [0], (7,8): [0], (3,0): [0], (6,3): [0], (4,1): [0], (7,4): [0], (5,2): [0], (8,5): [0], (7,5): [0], (7,2): [0], (4,2): [0], (6,4): [0], (3,1): [0], (6,1): [0]}
    Obs[(7,8)] = [1]
    Obs[(8,5)] = [1]
    Obs[(5,2)] = [1]
    Obs[(4,5)] = [1]
    Obs[(7,5)] = [1]
    Obs[(7,2)] = [1]
    Obs[(4,2)] = [1]
    E_X = [(7,5), (4,2), (7,2), (3,1)]

    logging.write("\r\n\r\nExample 5: ##########################")
    logging.write("No edges should be congested")
    logging.write("init_alpha_beta = (1, 1), which means equal chance to be congested or uncongested!\r\n ")
    Omega = {}
    for e in E:
        if np.mean(Obs[e]) > 0.5:
            Omega[e] = (10,1)
        else:
            Omega[e] = (1, 10)
    evaluate(V, E, Obs, Omega, E_X, logging, method, psl = False, approx = False, init_alpha_beta = (1, 1), report_stat = True)

    # return



    """
    0-> 1 -> 2
    ^   ^    ^
    3-> 4 -> 5
    ^   ^    ^
    6-> 7 -> 8
    """
    Obs = {(0,1): [0], (1,2): [0], (3,4): [0], (4,5): [0], (6,7): [0], (7,8): [0], (3,0): [0], (6,3): [0], (4,1): [0], (7,4): [0], (5,2): [0], (8,5): [0], (7,5): [0], (7,2): [0], (4,2): [0], (6,4): [0], (3,1): [0], (6,1): [0]}
    Obs[(7,4)] = [1]
    Obs[(4,1)] = [1]
    Obs[(1,2)] = [1]
    Obs[(4,5)] = [0]
    Obs[(7,5)] = [0]
    Obs[(7,2)] = [1]
    Obs[(4,2)] = [1]
    E_X = [(7,5), (4,2), (7,2), (3,1)]

    logging.write("\r\n\r\nExample 6: ##########################")
    logging.write("No edges should be congested")
    logging.write("init_alpha_beta = (1, 1), which means equal chance to be congested or uncongested!\r\n ")
    Omega = {}
    for e in E:
        if np.mean(Obs[e]) > 0.5:
            Omega[e] = (10,1)
        else:
            Omega[e] = (1, 10)
    evaluate(V, E, Obs, Omega, E_X, logging, method, psl = False, approx = False, init_alpha_beta = (1, 1), report_stat = True)

    # return

    """
    0-> 1 -> 2
    ^   ^    ^
    3-> 4 -> 5
    ^   ^    ^
    6-> 7 -> 8
    """
    Obs = {(0,1): [0], (1,2): [0], (3,4): [0], (4,5): [0], (6,7): [0], (7,8): [0], (3,0): [0], (6,3): [0], (4,1): [0], (7,4): [0], (5,2): [0], (8,5): [0], (7,5): [0], (7,2): [0], (4,2): [0], (6,4): [0], (3,1): [0], (6,1): [0]}
    Obs[(7,4)] = [1]
    Obs[(4,1)] = [1]
    Obs[(1,2)] = [1]
    Obs[(4,5)] = [1]
    Obs[(7,5)] = [1]
    Obs[(7,2)] = [1]
    Obs[(4,2)] = [1]
    E_X = [(7,5), (4,2), (7,2), (3,1)]

    logging.write("\r\n\r\nExample 7: ##########################")
    logging.write("No edges should be congested")
    logging.write("init_alpha_beta = (1, 1), which means equal chance to be congested or uncongested!\r\n ")
    Omega = {}
    for e in E:
        if np.mean(Obs[e]) > 0.5:
            Omega[e] = (10,1)
        else:
            Omega[e] = (1, 10)
    evaluate(V, E, Obs, Omega, E_X, logging, method, psl = False, approx = False, init_alpha_beta = (1, 1), report_stat = True)

    # return


    """
    0-> 1 -> 2
    ^   ^    ^
    3-> 4 -> 5
    ^   ^    ^
    6-> 7 -> 8
    """
    Obs = {(0,1): [0], (1,2): [0], (3,4): [0], (4,5): [0], (6,7): [0], (7,8): [0], (3,0): [0], (6,3): [0], (4,1): [0], (7,4): [0], (5,2): [0], (8,5): [0], (7,5): [0], (7,2): [0], (4,2): [0], (6,4): [0], (3,1): [0], (6,1): [0]}
    Obs[(7,4)] = [1]
    Obs[(4,1)] = [1]
    Obs[(1,2)] = [1]
    Obs[(4,5)] = [1]
    Obs[(0,1)] = [1]
    Obs[(7,5)] = [1]
    Obs[(7,2)] = [1]
    Obs[(4,2)] = [1]
    E_X = [(7,5), (4,2), (7,2), (3,1)]

    logging.write("\r\n\r\nExample 8: ##########################")
    logging.write("No edges should be congested")
    logging.write("init_alpha_beta = (1, 1), which means equal chance to be congested or uncongested!\r\n ")
    Omega = {}
    for e in E:
        if np.mean(Obs[e]) > 0.5:
            Omega[e] = (10,1)
        else:
            Omega[e] = (1, 10)
    evaluate(V, E, Obs, Omega, E_X, logging, method, psl = False, approx = False, init_alpha_beta = (1, 1), report_stat = True)

    # return

    """
    0-> 1 -> 2
    ^   ^    ^
    3-> 4 -> 5
    ^   ^    ^
    6-> 7 -> 8
    """
    Obs = {(0,1): [0], (1,2): [0], (3,4): [0], (4,5): [0], (6,7): [0], (7,8): [0], (3,0): [0], (6,3): [0], (4,1): [0], (7,4): [0], (5,2): [0], (8,5): [0], (7,5): [0], (7,2): [0], (4,2): [0], (6,4): [0], (3,1): [0], (6,1): [0]}
    Obs[(7,4)] = [1]
    Obs[(4,1)] = [1]
    Obs[(1,2)] = [1]
    Obs[(4,5)] = [1]
    Obs[(0,1)] = [1]
    Obs[(3,0)] = [1]
    Obs[(7,5)] = [1]
    Obs[(7,2)] = [1]
    Obs[(4,2)] = [1]
    Obs[(3,1)] = [1]
    E_X = [(7,5), (4,2), (7,2), (3,1)]

    logging.write("\r\n\r\nExample 9: ##########################")
    logging.write("No edges should be congested")
    logging.write("init_alpha_beta = (1, 1), which means equal chance to be congested or uncongested!\r\n ")
    Omega = {}
    for e in E:
        if np.mean(Obs[e]) > 0.5:
            Omega[e] = (10,1)
        else:
            Omega[e] = (1, 10)
    evaluate(V, E, Obs, Omega, E_X, logging, method, psl = False, approx = False, init_alpha_beta = (1, 1), report_stat = True)

    return


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
    experiment_proc_server()


def experiment_proc():
    logging = Log()
    methods = ["csl", "csl-3-rules", "csl-3-rules-conflict-evidence", "sl", "base1", "base2", "base3"][:3]
    graph_sizes = [500, 1000, 5000, 10000, 47676]
    # graph_sizes = [2500, 7500]
    ratios = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    realizations = 10
    for T in [6, 8, 11][1:]:
        for swap_ratio in [0.00, 0.01, 0.05][:1]:
            for test_ratio in [0.1, 0.2, 0.3, 0.4, 0.5][:]:                #percentage of edges to test (|E_x|/|E|)
                for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6][3:5]:      #the percentage of edges set the observations to 1
                    for ratio_conflict in [0.1, 0.2, 0.3, 0.4, 0.5,0.6][:]:
                        for graph_size in graph_sizes[2:3]:
                            for real_i in range(realizations)[:]:
                                logging.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                                for method in methods[:3]:
                                    f = "/home/apdm02/workspace/git/data/cls_conflict/trust-analysis3/nodes-{}-T-{}-rate-{}-testratio-{}-swaprate-{}-confictratio-{}-realization-{}-data-X.pkl".format(graph_size, T, ratio, test_ratio, swap_ratio, ratio_conflict, real_i)
                                    outf = '../output/test/{}_results-pos-{}-new5000.json'.format(method,ratio)
                                    # if not os.path.exists(f):
                                    #     print f
                                    #     continue
                                    #if os.path.exists(outf):
                                    #    continue

                                    logging.write("method: {}, T, {}, real_i: {}, ratio: {}, test_ratio: {}, swaprate: {}, graph_size: {}".format(method, T, real_i, ratio, test_ratio, swap_ratio, graph_size))
                                    logging.write(f)
                                    pkl_file = open(f, 'rb')
                                    [V, E, Obs, E_X, X_b] = pickle.load(pkl_file)
                                    E_X=dict(itertools.izip_longest(*[iter(E_X)] * 2, fillvalue=""))
                                    n = len(V)
                                    n_E = len(E)
                                    ndays = len(Obs[E[0]])
                                    # print ndays
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
                                        Omega = calc_Omega_from_Obs(t_Obs, E)
                                        # print "Omega", Omega
                                        # for e, val in Omega.items():
                                        #     if val[0] > 0:
                                        #         print e, val
                                        # return
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
                                    logging.write("prob_mse: {}, running time: {}".format(mu_prob_mse, running_time))
                                   # info = "{}-{}-{}-{}, network size: {}, S: {}, T: {}, test ratio: {}, acc: ({}, {}), prob_mse: ({}, {}), time: {}".format(dataset, weekday, hour, ref_per, n, ndays - T + 1, T, test_ratio, "%.2f" % mu_accuracy, "%.2f" % sigma_accuracy, "%.2f" % mu_prob_mse, "%.2f" % sigma_prob_mse, running_time)
                                    # print info
                                    # logging.write(info)
                                    result_ = {'network_size': graph_size, 'positive_ratio': ratio, "realization": real_i, 'sample_size': ndays - T + 1, 'T': T,'ratio_conflict':ratio_conflict,
                                            'test_ratio': test_ratio, 'acc': (mu_accuracy, sigma_accuracy), 'alpha_mse': (mu_alpha_mse, sigma_alpha_mse), 'beta_mse': (mu_beta_mse, sigma_beta_mse), 'u_mse': (mu_u_mse, sigma_u_mse), 'b_mse': (mu_b_mse, sigma_b_mse), 'd_mse': (mu_d_mse, sigma_d_mse), 'prob_mse': (mu_prob_mse, sigma_prob_mse), 'runtime': running_time}
                                    # print result_
                                    outfp = open(outf, 'a')
                                    outfp.write(json.dumps(result_) + '\n')
                                    outfp.close()

def experiment_proc_server():
    logging = Log()
    data_root = "/network/rit/lab/ceashpc/adil/data/csl-data/apr21/"
    methods = ["csl", "csl-3-rules", "csl-3-rules-conflict-evidence", "sl", "base1", "base2", "base3"][:3]
    # graph_sizes = [500, 1000, 5000, 10000, 47676]
    graph_sizes = [1000, 5000,47676]
    ratios = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    realizations = 1
    case_count=0
    for graph_size in graph_sizes[2:]:
        for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6][:3]:#the percentage of edges set the observations to 1
            for T in [10,20][:1]:
                for swap_ratio in [0.00, 0.01, 0.05][:1]:
                    for test_ratio in [0.1, 0.2, 0.3, 0.4, 0.5][:]:                #percentage of edges to test (|E_x|/|E|)
                        for ratio_conflict in [0.1, 0.2, 0.3, 0.4, 0.5,0.6][:]:
                            for real_i in range(realizations)[:]:
                                out_folder = data_root + str(graph_size) + "/"
                                logging.write(str(case_count)+" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                                case_count+=1.0
                                for method in methods[:]:
                                    f = out_folder + "nodes-{}-T-{}-rate-{}-testratio-{}-swaprate-{}-confictratio-{}-realization-{}-data-X.pkl".format(
                                        graph_size, T, ratio, test_ratio, swap_ratio, ratio_conflict, real_i)
                                    outf = '../output/test/{}_results-server-Apr26-time.json'.format(method,ratio)
                                    # if not os.path.exists(f):
                                    #     print f
                                    #     continue
                                    #if os.path.exists(outf):
                                    #    continue

                                    logging.write("method: {}, T, {}, real_i: {}, ratio: {}, test_ratio: {}, swaprate: {}, graph_size: {}".format(method, T, real_i, ratio, test_ratio, swap_ratio, graph_size))
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
                                        Omega = calc_Omega_from_Obs(t_Obs, E)
                                        # print "Omega", Omega
                                        # for e, val in Omega.items():
                                        #     if val[0] > 0:
                                        #         print e, val
                                        # return
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
                                    logging.write("prob_mse: {}, running time: {}".format(mu_prob_mse, running_time))
                                   # info = "{}-{}-{}-{}, network size: {}, S: {}, T: {}, test ratio: {}, acc: ({}, {}), prob_mse: ({}, {}), time: {}".format(dataset, weekday, hour, ref_per, n, ndays - T + 1, T, test_ratio, "%.2f" % mu_accuracy, "%.2f" % sigma_accuracy, "%.2f" % mu_prob_mse, "%.2f" % sigma_prob_mse, running_time)
                                    # print info
                                    # logging.write(info)
                                    result_ = {'network_size': graph_size, 'positive_ratio': ratio, "realization": real_i, 'sample_size': ndays - T + 1, 'T': T,'ratio_conflict':ratio_conflict,
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
    if method == 'sl':
        pred_omega_x = SL_prediction(V, E, Obs, Omega, copy.deepcopy(E_X))
        # print pred_omega_x
    elif method == 'csl':
        pred_omega_x = inference_apdm_format(V, E, Obs, Omega, E_X, logging)
    elif method == 'csl-3-rules':
        # b = {e: 0 for e in E}
        b={}
        # X_b = []
        X_b = {e: 0 for e in E if not E_X.has_key(e)}
        psl = True
        pred_omega_x, _ = inference_apdm_format_conflict_evidence(V, E, Obs, Omega, b, X_b, E_X, logging, psl)
    elif method == 'csl-3-rules-conflict-evidence':
        # b = {e: 0 for e in E}
        b={}
        # X_b = {e: 0 for e in E if e not in E_X}
        X_b = {e: 0 for e in E if not E_X.has_key(e)}
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

