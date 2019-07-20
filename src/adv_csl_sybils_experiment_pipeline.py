__author__ = ''
from math import *
import numpy as np
import copy
import random
import os
import pickle
import sys
import multiprocessing
import time
from basic_funs import *
from network_funs import *
# from simulator import *
from scipy.stats import beta
# from writeAPDM import writeAPDM
# from readAPDM import readAPDM
import scipy
# scipy.stats.
# import scipy.stats
from scipy.stats import beta
from log import Log
import time
time.time()
from feng_SL_inference_multiCore_node2 import *
import json
#from feng_SL_inference import *
from multi_core_csl_inference_adversarial_sybils import inference_apdm_format as inference_apdm_format_conflict_evidence
from multi_core_csl_plus_sybils_inference import inference_apdm_format as csl_plus_inference



def baseline(V, E, Obs, Omega, E_X):
    # np.random.seed(123)
    op={0:(1,1),1:(0.001,11),2:(11,0.001)}
    Omega_X = {}
    for e in E_X:
        Omega_X[e] = op[np.random.choice([1,2])]
    return Omega_X

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

class Task_inference(object):
    def __init__(self, omega, y_t, Y, X, node_nns, p0, R, psl, approx, report_stat):
        self.omega = omega
        self.y_t = y_t
        self.Y = Y
        self.X = X
        self.node_nns = node_nns
        self.p0 = p0
        self.R = R
        self.psl = psl
        self.approx = approx
        self.report_stat = report_stat

    def __call__(self):
        # this is the place to do your work
        # time.sleep(0.1) # pretend to take some time to do our work
                      # omega, y_t, Y, X, edge_up_nns, edge_down_nns, p0, R, psl, approx, report_stat
        p_t = admm(self.omega, self.y_t, self.Y, self.X, self.node_nns,self.p0, self.R, self.psl, self.approx, self.report_stat)
        return p_t

    def __str__(self):
        return '%s' % (self.p0)


def calc_initial_p1_nodes0(y_t, node_nns, X, Y, cnt_V, p0):
    p = [0 for i in range(cnt_V)]
    for v, v_nns in node_nns.items():
        if X.has_key(v):
            obs = {v_n:y_t[v_n] for v_n in v_nns if Y.has_key(v_n)}
            # if len(obs) == 0: obs = {p0}
            conf = 0
            n_pos=0.0
            for v_o,val in obs.items():
                n_pos+=val
                if val >0.0:
                    conf += 1.0
                else:
                    conf -= 1.0
            if conf > 1.0:
                p[v] = 1.0
            # else:
            #     if n_pos<=1.0:
            #         p[e] = 0.0
            #     else:
            #         p[e] = 1.0
            # p[e] = np.median(obs)
        else:
            p[v] = y_t[v]
    return p


"""
INPUT
V = [0, 1, ...] is a list of vertex ids
E: a list of pairs of vertex ids
Obs: a dictionary with key edge and its value a list of congestion observations of this edge from t = 1 to t = T
Omega: a dictionary with key edge and its value a pair of alpha and beta.
E_X: Set of edges (pairs of nodes)
logging: log object for loging intermediate information
psl: True if we want to use pure PSL inference, instead of the inference based on the proposed model; False, otherwise.
approx: True if we want to try the probability values {0.01, 0.02, ..., 0.99} to solve each cubic function equation; False and calculate the exact solution, otherwise.
init_alpha_beta: the initial pair of alpha and beta for the edges in E_X
report_stat: If True, then intermediate information will be logged.
OUTPUT
omega_x: a dictionary with key edge pair and value a pair of alpha and beta values
"""
def inference_apdm_format(V, E, Obs, Omega, E_X, logging, psl = False, approx = True, init_alpha_beta = (1, 1), report_stat = False):
    if report_stat: print "start reformat"
    node_nns, id_2_node, node_2_id, omega, feat = reformat_nodes(V, E, Obs, Omega)
    # print id_2_edge[648]
    omega_y = {}
    p0 = 0.5
    y = {}
    X = {node_2_id[v]: 1 for v in E_X}
    Y = {v: 1 for v in id_2_node.keys() if v not in X}
    for v in Y.keys():
        omega_y[v] = omega[v]
        y[v] = feat[v]
    T = len(feat[v])
    if report_stat: print "number of time stamps: {}".format(T)
    x = {}
    for v in X.keys():
        x[v] = feat[v]
    if report_stat: print "start generate_PSL_rules_from_edge_cnns"
    R = generate_PSL_rules_from_edge_cnns_nodes(node_nns)
    print "#Rule:",len(R)
    if report_stat: print "start inference"

    for v in X.keys(): omega[v] = init_alpha_beta # Beta pdf can be visualized via http://eurekastatistics.com/beta-distribution-pdf-grapher/
    epsilon = 0.01
    maxiter = 5
    for iter in range(maxiter):
        p = []
        tasks = multiprocessing.Queue()
        results = multiprocessing.Queue()

        # Start consumers
        num_consumers = T
        # print iter, 'Creating %d consumers' % num_consumers
        consumers = [Consumer(tasks, results)
                     for i in xrange(num_consumers)]
        for w in consumers:
            w.start()

        num_jobs = T

        for t in range(T):
            y_t = {v: v_y[t] for v, v_y in y.items()}
            # x_t = {e: e_x[t] for e, e_x in x.items()}
            tasks.put(Task_inference(omega, y_t, Y, X, node_nns, p0, R, psl, approx, report_stat))

        # Add a poison pill for each consumer
        for i in xrange(num_consumers):
            tasks.put(None)

        fin = 0
        # Start printing results
        while num_jobs:
            p_t = results.get()
            p.append(p_t)
            num_jobs -= 1

        error = 0.0
        omega_x_prev = {v: alpha_beta for v, alpha_beta in omega.items() if v in X}
        omega = estimate_omega_x(p, X)
        omega_x = {v: alpha_beta for v, alpha_beta in omega.items() if v in X}
        for v in X:
            alpha_prev, beta_prev = omega_x_prev[v]
            alpha, beta = omega_x[v]
            error += pow(alpha_prev-alpha, 2) + pow(beta_prev-beta, 2)
        error = sqrt(error)
        if error < 0.01:
            break
    omega_x = {id_2_node[v]: alpha_beta for v, alpha_beta in omega_x.items()}
    W = 2.0
    if report_stat:
        for e in E_X:
            logging.write("************ edge vertex ids: {0}, ".format(e))
            logging.write("---- omega_x {0}: {1:.2f}, {2:.2f}".format(Omega[e], omega_x[e][0], omega_x[e][1]))
            logging.write("---- uncertainty ({0}): {1}".format(W/(Omega[e][0] + Omega[e][1]), W/(omega_x[e][0] + omega_x[e][1])))
    return omega_x






"""
INPUT
omega: A dictionary of opinions (tuples of alpha and beta values) of all edges (X+Y): [edge1 id: [alpha_1, beta_1], ...]. The opinions of edges in X are the current estimates.
y_t: A dictionary of lists of observations at time t with key an edge id and value its observation at time t.
Y: The subset of edge ids that have known opinions
X: The subset of edge ids whose opinions will be predicted.
nns: A dictionary with key edge, and value the list of neighboring edge ids
edge_nns: key is an edge id, and its value is a list of its neighbor edge ids.
p0: The prior probability that an edge is congested. It is calculated the ratio of total number of congested links indexed by link id and time over the total number of links.
R: list of neighboring pairs of edges: [(edge1, edge2), ...]. Each pair has two related PSL rules: edge1 -> edge2 and edge2 -> edge1
psl: True if we want to use pure PSL inference, instead of the inference based on the proposed model; False, otherwise.
approx: True if we want to try the probability values {0.01, 0.02, ..., 0.99} to solve each cubic function equation; False and calculate the exact solution, otherwise.

OUTPUT
p: a list of probability values of the cnt_Y + cnt_X edges. p[i] refers to the probability value of edge id=i
"""
def admm(omega, y_t, Y, X, node_nns, p0, R, psl = False, approx = False, report_stat = False):
    weight = 1.0
    epsilon = 0.01
    cnt_Y = len(Y)
    cnt_X = len(X)
    cnt_V = cnt_X + cnt_Y
    K = len(R)
    p = calc_initial_p1_nodes0(y_t, node_nns, X, Y, cnt_V, p0)
    R_p = []
    R_z = [] # R_z is a vector of copied variables in R_p. R_z[e] is a copied variable of R_p[e]
    R_lambda_ = []
    copies = {}
    for k in range(K):
        rule_z = []
        rule_p = []
        rule_lambda_ = []
        for idx, v in enumerate(R[k]):
            rule_z.append(p[v])
            rule_p.append(p[v])
            rule_lambda_.append(0.0)
            if copies.has_key(v):
                copies[v].append([k, idx])
            else:
                copies[v] = [[k, idx]]
        R_z.append(rule_z)
        R_p.append(rule_p)
        R_lambda_.append(rule_lambda_)
    # rho = 0.0001 # Set rho to be quite low to start with
    rho = 1.0
    maxiter = 5
    # maxRho = 5
    for iter in range(maxiter):
        for k in range(K):
            # update lambda variables
            R_lambda_[k] = list_sum(R_lambda_[k], list_dot_scalar(list_minus(R_z[k], R_p[k]), rho))
            # update copy variables
            R_z[k] = normalize(list_minus(R_p[k], list_dot_scalar(R_lambda_[k], -1.0 / rho)))
            if R_z[k][0] > R_z[k][1]:
                """
                The minimization problem to be solved is:
                    min weight * [1, -1] * zxy[k] + rho/2 || zxy[k] - pxy[k] + 1/rho * lambda_xy[k] ||_2^2

                The gradient of the objective function over zxy[k] is:
                    weight * [1, -1] + rho (zxy[k] - pxy[k] + 1/rho * lambda_xy[k])

                The updates zxy[k] can be identified such that the above gradient equals to 0:
                    zxy[k] = pxy[k] - 1/rho * lambda_xy[k] - weight * 1/rho [1, -1]
                """
                R_z[k] = normalize(list_minus(R_p[k], list_sum(list_dot_scalar(R_lambda_[k], 1/rho), [weight/rho, -1 * weight/rho])))
                if R_z[k][0] < R_z[k][1]:
                    R_z[k][0] = R_z[k][1] = np.mean(list_minus(R_p[k], list_dot_scalar(R_lambda_[k], 1.0/rho)))
                    if R_z[k][0] < 0:
                        R_z[k][0] = R_z[k][1] = 0
        # update probability variables
        p_old = R_p_2_p(R_p, copies, cnt_V)
        if psl == True:
            R_p = psl_update_p(Y, R_p, R_z, R_lambda_, copies, omega, cnt_V, rho)
        else:
            R_p = update_p(X, R_p, R_z, R_lambda_, copies, omega, cnt_V, rho, approx)
        # rho = min(maxRho, rho * 1.1)
        p = R_p_2_p(R_p, copies, cnt_V)
        error = sqrt(np.sum([pow(p_old[e] - p[e], 2) for e in range(cnt_V)]))
        # print ">>>>>>>>>>>> admm iteration.{0}: {1}".format(iter, error)
        if error < epsilon:
            break
    return p




"""
INPUT:
true_omega_x: The true opinions of edges in X
pred_omega_x: The predicted opinions of edges in X
X: the list of edges whose opinions are predicted
logging: file object for logging

OUTPUT
prob_mse:
u_mse:
prob_relative_mse:
u_relative_mse:
accuracy:
recall_congested:
recall_uncongested:
"""
def calculate_measures(true_omega_x, pred_omega_x, X, logging):
    # print "X", X
    W = 2.0
    bs = []
    ds = []
    for e in X:
        b1,d1,u1,a1 = beta_to_opinion2(true_omega_x[e][0], true_omega_x[e][1])
        b2,d2,u2,a2 = beta_to_opinion2(pred_omega_x[e][0], pred_omega_x[e][1])
        bs.append(np.abs(b1-b2))
        ds.append(np.abs(d1-d2))
    b_mse = np.mean(bs)
    d_mse = np.mean(ds)
    alpha_mse = np.mean([np.abs(true_omega_x[e][0] - pred_omega_x[e][0]) for e in X])
    beta_mse = np.mean([np.abs(true_omega_x[e][1] - pred_omega_x[e][1]) for e in X])
    u_true_X = {e: np.abs((W*1.0)/(true_omega_x[e][0] + true_omega_x[e][1] + W)) for e in X}
    u_pred_X = {e: np.abs((W*1.0)/(pred_omega_x[e][0] + pred_omega_x[e][1] + W)) for e in X}
    # print "u_true_X", u_true_X
    # print "u_pred_X", u_pred_X
    u_mse = np.mean([np.abs(u_pred_X[e] - u_true_X[e]) for e in X])
    u_relative_mse = np.mean([abs(u_pred_X[e] - u_true_X[e])/u_true_X[e] for e in X])
    prob_true_X = {e: (true_omega_x[e][0]*1.0)/(true_omega_x[e][0] + true_omega_x[e][1])+0.0001 for e in X}
    prob_pred_X = {e: (pred_omega_x[e][0]*1.0)/(pred_omega_x[e][0] + pred_omega_x[e][1]) for e in X}
    # for e in X:
    #     print
    # print "prob_true_X", prob_true_X
    # print "prob_pred_X", prob_pred_X
    # print [np.abs(prob_pred_X[e] - prob_true_X[e]) for e in X], np.mean([np.abs(prob_pred_X[e] - prob_true_X[e]) for e in X])
    prob_mse = np.mean([np.abs(prob_pred_X[e] - prob_true_X[e]) for e in X])
    prob_relative_mse = np.mean([np.abs(prob_pred_X[e] - prob_true_X[e])/prob_true_X[e] for e in X])
    # logging.write("********************* uncertainty MSE: {0}, RMSE: {1}".format(u_mse, u_relative_mse))
    # logging.write("********************* probability MSE: {0}, RMSE: {1}".format(prob_mse, prob_relative_mse))
    recall_congested = 0.0
    n_congested = 0.01
    recall_uncongested = 0.0
    n_uncongested = 0.01
    for e in X:
        if prob_true_X[e] >= 0.5:
            n_congested += 1
            if prob_pred_X[e] >= 0.5:
                recall_congested += 1
        else:
            n_uncongested += 1
            if prob_pred_X[e] < 0.5:
                recall_uncongested += 1
    accuracy = (recall_congested + recall_uncongested) * 1.0 / (n_congested + n_uncongested)
    if recall_congested > 0:
        recall_congested = recall_congested / n_congested
    else:
        recall_congested = -1
    if recall_uncongested > 0:
        recall_uncongested = recall_uncongested / n_uncongested
    else:
        recall_uncongested = -1
    # logging.write("********************* accuracy: {0}, recall congested: {1}, recall uncongested: {2}".format(accuracy, recall_congested, recall_uncongested))
    return alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested


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
"""
def  evaluate(V, E, Obs, Omega, E_X, logging, method = 'sl', psl = False, approx = False, init_alpha_beta = (1, 1), report_stat = False):
    running_starttime = time.time()
    if method == 'SL':
        pred_omega_x = SL_prediction_multiCore_node(V, E, Obs, Omega, copy.deepcopy(E_X))
    elif method == 'CSL':
        pred_omega_x = inference_apdm_format(V, E, Obs, Omega, E_X, logging)
    elif method == 'csl-conflict-1':
        # obs_dict = {}
        # for e, ob in Obs.items():
        #     if obs_dict.has_key(sum(Obs[e])):
        #         obs_dict[sum(Obs[e])] += 1
        #     else:
        #         obs_dict[sum(Obs[e])] = 1
        # print(obs_dict)
        # b = {e: 0 for e in E}
        b = {}
        psl = True
        #V, E, Obs, Omega, b, X, logging, psl = False, approx = True, init_alpha_beta = (1, 1), report_stat = False
        pred_omega_x, _ = inference_apdm_format_conflict_evidence(V, E, Obs, Omega, b, E_X, logging, psl)
    elif method == 'Adv-CSL':
        # b = {e: 0 for e in E}
        b = {}
        psl = False
        pred_omega_x, _ = inference_apdm_format_conflict_evidence(V, E, Obs, Omega, b, E_X, logging, psl)
    elif method == 'CSL-Plus':
        # b = {e: 0 for e in E}
        b = {}
        psl = False
        pred_omega_x, _ = csl_plus_inference(V, E, Obs, Omega, b, E_X, logging, psl)

    elif method == 'Baseline':
        pred_omega_x = baseline(V, E, Obs, Omega, E_X)
    else:
        raise Exception("Method Error")
    alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested = calculate_measures(Omega, pred_omega_x, E_X, logging)
    running_endtime = time.time()
    running_time = running_endtime - running_starttime
    return alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested, running_time


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
Evaluate the performances of the testing methods on real-world datasets.
"""
def facebook_sybils_dataset_test():
    logging = Log()
    dataroot = "/network/rit/lab/ceashpc/adil/data/adv_csl/Jan2/"
    report_stat = False
    count=0
    realizations=1
    gammas=[0.0, 0.01, 0.03, 0.05, 0.07,0.09,0.2,0.3,0.4]
    # gammas=[0.0, 0.01,0.3,0.4]
    methods = ["SL","CSL", "Adv-CSL","Baseline","CSL-Plus"][3:4]
    for real_i in range(realizations)[:1]:
        for test_ratio in [0.3,0.1, 0.2, 0.4, 0.5][:1]:
            for adv_type in ["random_pgd","random_noise","random_pgd_csl","random_pgd_gcn_vae"][:1]:
                for attack_edge in [10000,35000][:1]:
                    for T in [10][:]:
                        for swap_ratio in [0.00, 0.01, 0.02, 0.05][1:2]:
                            for gamma in gammas:  # 11
                                logging.write(str(count)+" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`")
                                count+=1.0
                                for method in methods[:]:
                                    f=dataroot +adv_type+ "/facebook/facebook-attackedges-{}-T-{}-testratio-{}-swap_ratio-{}-gamma-{}-realization-{}-data-X.pkl".format(attack_edge, T, test_ratio,swap_ratio, gamma, real_i)
                                    outf = '../output/sybils/{}_results-server-July20-{}.json'.format(method,adv_type)
                                    logging.write("dataset: {} method: {}, #attack_edge:{},T:{},test_ratio: {},gamma:{}".format("facebook",method,attack_edge,T,test_ratio,gamma))
                                    logging.write(f)
                                    pkl_file = open(f, 'rb')
                                    [V, E, Obs, E_X, X_b] = pickle.load(pkl_file)
                                    n = len(V)
                                    n_E = len(E)
                                    ndays = len(Obs.values()[0])
                                    T=ndays
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
                                    # m_idx=int(round(T/2.0))
                                    for start_t in range(ndays - T + 1)[:1]:
                                        # print "start_t", start_t
                                        # t_Obs = {e: e_Obs[m_idx-5:m_idx+6] for e, e_Obs in Obs.items()}
                                        t_Obs = {v: v_Obs[:] for v, v_Obs in Obs.items()}
                                        Omega = calc_Omega_from_Obs2(t_Obs, V)
                                        alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested, running_time = evaluate(V, E, t_Obs, Omega, E_X, logging, method)
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
                                    result_ = {'dataset':"facebook",'attack_edge':attack_edge,'network_size': n,'adv_type':adv_type,
                                               'sample_size': ndays - T + 1, 'T': T, 'gamma': gamma,
                                               'test_ratio': test_ratio,'swap_ratio':swap_ratio, 'acc': (mu_accuracy, sigma_accuracy),
                                               'prob_mse': (mu_prob_mse, sigma_prob_mse),'alpha_mse': (mu_alpha_mse, sigma_alpha_mse), 'beta_mse': (mu_beta_mse, sigma_beta_mse), 'u_mse': (mu_u_mse, sigma_u_mse), 'b_mse': (mu_b_mse, sigma_b_mse), 'd_mse': (mu_d_mse, sigma_d_mse),'realization':real_i, 'runtime': running_time}
                                    outfp = open(outf, 'a')
                                    outfp.write(json.dumps(result_) + '\n')
                                    outfp.close()


def enron_sybils_dataset_test():
    logging = Log()
    dataroot = "/network/rit/lab/ceashpc/adil/data/adv_csl/Jan2/"
    report_stat = False
    count=0
    realizations=10

    for test_ratio in [0.3,0.1, 0.2, 0.4, 0.5][:1]:
        methods = ["SL","CSL", "Adv-CSL","Baseline","CSL-Plus"][3:4]
        for adv_type in ["random_noise","random_pgd_csl","random_pgd_gcn_vae","random_pgd"][3:]:
            for attack_edge in [1000,5000,10000,15000,20000][2:3]:
                for T in [10][:]:
                    for swap_ratio in [0.00, 0.01, 0.02, 0.05][1:2]:
                        for gamma in [0.0, 0.01, 0.03, 0.05, 0.07,0.09,0.2,0.3,0.4][:]:  # 11
                            for real_i in range(realizations)[:1]:
                                logging.write(str(count)+" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`")
                                count+=1.0
                                for method in methods[:]:
                                    f=dataroot +adv_type+ "/enron/enron-attackedges-{}-T-{}-testratio-{}-swap_ratio-{}-gamma-{}-realization-{}-data-X.pkl".format(attack_edge, T, test_ratio,swap_ratio, gamma, real_i)
                                    outf = '../output/sybils/{}_results-server-July20-{}.json'.format(method,adv_type)
                                    logging.write("dataset: {} method: {}, #attack_edge:{},T:{},test_ratio: {},gamma:{}".format("enron",method,attack_edge,T,test_ratio,gamma))
                                    logging.write(f)
                                    pkl_file = open(f, 'rb')
                                    [V, E, Obs, E_X, X_b] = pickle.load(pkl_file)
                                    n = len(V)
                                    n_E = len(E)
                                    ndays = len(Obs.values()[0])
                                    T=ndays
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
                                    # m_idx=int(round(T/2.0))
                                    for start_t in range(ndays - T + 1)[:1]:
                                        # print "start_t", start_t
                                        # t_Obs = {e: e_Obs[m_idx-5:m_idx+6] for e, e_Obs in Obs.items()}
                                        t_Obs = {v: v_Obs[:] for v, v_Obs in Obs.items()}
                                        Omega = calc_Omega_from_Obs2(t_Obs, V)
                                        alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested, running_time = evaluate(V, E, t_Obs, Omega, E_X, logging, method)
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
                                    result_ = {'dataset':"enron",'attack_edge':attack_edge,'network_size': n,'adv_type':adv_type,
                                               'sample_size': ndays - T + 1, 'T': T, 'gamma': gamma,
                                               'test_ratio': test_ratio,'swap_ratio':swap_ratio, 'acc': (mu_accuracy, sigma_accuracy),
                                               'prob_mse': (mu_prob_mse, sigma_prob_mse),'alpha_mse': (mu_alpha_mse, sigma_alpha_mse), 'beta_mse': (mu_beta_mse, sigma_beta_mse), 'u_mse': (mu_u_mse, sigma_u_mse), 'b_mse': (mu_b_mse, sigma_b_mse), 'd_mse': (mu_d_mse, sigma_d_mse),'realization':real_i, 'runtime': running_time}
                                    outfp = open(outf, 'a')
                                    outfp.write(json.dumps(result_) + '\n')
                                    outfp.close()


def slashdot_sybils_dataset_test():
    logging = Log()
    dataroot = "/network/rit/lab/ceashpc/adil/data/adv_csl/Jan2/"
    report_stat = False
    count=0
    realizations=10
    methods = ["SL","CSL", "Adv-CSL","Baseline","CSL-Plus"][3:4]
    for adv_type in ["random_noise","random_pgd","random_pgd_csl","random_pgd_gcn_vae"][1:2]:
        for attack_edge in [1000,5000,10000,15000,20000][2:3]:
            for T in [10][:]:
                for swap_ratio in [0.00, 0.01, 0.02, 0.05][1:2]:
                    for test_ratio in [0.1, 0.2,0.3,0.4, 0.5][2:3]:
                        for gamma in [0.0, 0.01, 0.03, 0.05, 0.07,0.09,0.2,0.3,0.4][:]:  # 11
                            for real_i in range(realizations)[:1]:
                                logging.write(str(count)+" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`")
                                count+=1.0
                                for method in methods[:]:
                                    f=dataroot +adv_type+ "/slashdot/slashdot-attackedges-{}-T-{}-testratio-{}-swap_ratio-{}-gamma-{}-realization-{}-data-X.pkl".format(attack_edge, T, test_ratio,swap_ratio, gamma, real_i)
                                    outf = '../output/sybils/{}_results-server-July20-{}.json'.format(method,adv_type)
                                    logging.write("dataset: {} method: {}, #attack_edge:{},T:{},test_ratio: {},gamma:{}".format("slashdot",method,attack_edge,T,test_ratio,gamma))
                                    logging.write(f)
                                    pkl_file = open(f, 'rb')
                                    [V, E, Obs, E_X, X_b] = pickle.load(pkl_file)
                                    n = len(V)
                                    n_E = len(E)
                                    ndays = len(Obs.values()[0])
                                    T=ndays
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
                                    # m_idx=int(round(T/2.0))
                                    for start_t in range(ndays - T + 1)[:1]:
                                        # print "start_t", start_t
                                        # t_Obs = {e: e_Obs[m_idx-5:m_idx+6] for e, e_Obs in Obs.items()}
                                        t_Obs = {v: v_Obs[:] for v, v_Obs in Obs.items()}
                                        Omega = calc_Omega_from_Obs2(t_Obs, V)
                                        alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested, running_time = evaluate(V, E, t_Obs, Omega, E_X, logging, method)
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
                                    result_ = {'dataset':"slashdot",'attack_edge':attack_edge,'network_size': n,'adv_type':adv_type,
                                               'sample_size': ndays - T + 1, 'T': T, 'gamma': gamma,
                                               'test_ratio': test_ratio,'swap_ratio':swap_ratio, 'acc': (mu_accuracy, sigma_accuracy),
                                               'prob_mse': (mu_prob_mse, sigma_prob_mse),'alpha_mse': (mu_alpha_mse, sigma_alpha_mse), 'beta_mse': (mu_beta_mse, sigma_beta_mse), 'u_mse': (mu_u_mse, sigma_u_mse), 'b_mse': (mu_b_mse, sigma_b_mse), 'd_mse': (mu_d_mse, sigma_d_mse),'realization':real_i, 'runtime': running_time}
                                    outfp = open(outf, 'a')
                                    outfp.write(json.dumps(result_) + '\n')
                                    outfp.close()


def main():
    facebook_sybils_dataset_test()
    enron_sybils_dataset_test()
    slashdot_sybils_dataset_test()

if __name__=='__main__':
    main()