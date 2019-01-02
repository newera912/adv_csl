__author__ = 'Feng Chen'
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
import baseline
from feng_SL_inference_multiCore import *
import json
#from feng_SL_inference import *
from multi_core_csl_inference_conflicting_evidence_traffic import inference_apdm_format as inference_apdm_format_conflict_evidence
# import networkx as nx
# import matplotlib.pyplot as plt

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

    def __init__(self, omega, y_t, Y, X, edge_up_nns, edge_down_nns, p0, R, psl, approx, report_stat):
        self.omega = omega
        self.y_t = y_t
        self.Y = Y
        self.X = X
        self.edge_up_nns = edge_up_nns
        self.edge_down_nns = edge_down_nns
        self.p0 = p0
        self.R = R
        self.psl = psl
        self.approx = approx
        self.report_stat = report_stat

    def __call__(self):
        # this is the place to do your work
        # time.sleep(0.1) # pretend to take some time to do our work
                      # omega, y_t, Y, X, edge_up_nns, edge_down_nns, p0, R, psl, approx, report_stat
        p_t = admm(self.omega, self.y_t, self.Y, self.X, self.edge_up_nns,self. edge_down_nns, self.p0, self.R, self.psl, self.approx, self.report_stat)
        return p_t

    def __str__(self):
        return '%s' % (self.p0)

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
    edge_up_nns, edge_down_nns, id_2_edge, edge_2_id, omega, feat = reformat(V, E, Obs, Omega)
    # print id_2_edge[648]
    omega_y = {}
    p0 = 0.5
    y = {}
    X = {edge_2_id[e]: 1 for e in E_X}
    Y = {e: 1 for e in id_2_edge.keys() if e not in X}
    for e in Y.keys():
        omega_y[e] = omega[e]
        y[e] = feat[e]
    T = len(feat[e])
    if report_stat: print "number of time stamps: {}".format(T)
    x = {}
    for e in X.keys():
        x[e] = feat[e]
    if report_stat: print "start generate_PSL_rules_from_edge_cnns"
    R = generate_PSL_rules_from_edge_cnns(edge_up_nns)
    if report_stat: print "start inference"

    for e in X.keys(): omega[e] = init_alpha_beta # Beta pdf can be visualized via http://eurekastatistics.com/beta-distribution-pdf-grapher/
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
            y_t = {e: e_y[t] for e, e_y in y.items()}
            # x_t = {e: e_x[t] for e, e_x in x.items()}
            tasks.put(Task_inference(omega, y_t, Y, X, edge_up_nns, edge_down_nns, p0, R, psl, approx, report_stat))

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
        omega_x_prev = {e: alpha_beta for e, alpha_beta in omega.items() if e in X}
        omega = estimate_omega_x(p, X)
        omega_x = {e: alpha_beta for e, alpha_beta in omega.items() if e in X}
        for e in X:
            alpha_prev, beta_prev = omega_x_prev[e]
            alpha, beta = omega_x[e]
            error += pow(alpha_prev-alpha, 2) + pow(beta_prev-beta, 2)
        error = sqrt(error)
        if error < 0.01:
            break
    omega_x = {id_2_edge[e]: alpha_beta for e, alpha_beta in omega_x.items()}
    W = 2.0
    if report_stat:
        for e in E_X:
            logging.write("************ edge vertex ids: {0}, ".format(e))
            logging.write("---- omega_x {0}: {1:.2f}, {2:.2f}".format(Omega[e], omega_x[e][0], omega_x[e][1]))
            logging.write("---- uncertainty ({0}): {1}".format(W/(Omega[e][0] + Omega[e][1]), W/(omega_x[e][0] + omega_x[e][1])))
    return omega_x



def inference_apdm_format_single(V, E, Obs, Omega, E_X, logging, psl = False, approx = True, init_alpha_beta = (1, 1), report_stat = False):
    if report_stat: print "start reformat"
    edge_up_nns, edge_down_nns, id_2_edge, edge_2_id, omega, feat = reformat(V, E, Obs, Omega)
    # print id_2_edge[648]
    omega_y = {}
    p0 = 0.5
    y = {}
    X = {edge_2_id[e]: 1 for e in E_X}
    Y = {e: 1 for e in id_2_edge.keys() if e not in X}
    for e in Y.keys():
        omega_y[e] = omega[e]
        y[e] = feat[e]
    T = len(feat[e])
    if report_stat: print "number of time stamps: {}".format(T)
    x = {}
    for e in X.keys():
        x[e] = feat[e]
    if report_stat: print "start generate_PSL_rules_from_edge_cnns"
    R = generate_PSL_rules_from_edge_cnns(edge_up_nns)
    if report_stat: print "start inference"

    for e in X.keys(): omega[e] = init_alpha_beta # Beta pdf can be visualized via http://eurekastatistics.com/beta-distribution-pdf-grapher/
    epsilon = 0.01
    maxiter = 5
    for iter in range(maxiter):
        p = []

        for t in range(T):
            y_t = {e: e_y[t] for e, e_y in y.items()}
            # x_t = {e: e_x[t] for e, e_x in x.items()}
            #p_t = admm(self.omega, self.y_t, self.Y, self.X, self.edge_up_nns,self. edge_down_nns, self.p0, self.R, self.psl, self.approx, self.report_stat)
            p_t = admm(omega, y_t, Y, X, edge_up_nns, edge_down_nns, p0, R, psl, approx, report_stat)
            p.append(p_t)


        error = 0.0
        omega_x_prev = {e: alpha_beta for e, alpha_beta in omega.items() if e in X}
        omega = estimate_omega_x(p, X)
        omega_x = {e: alpha_beta for e, alpha_beta in omega.items() if e in X}
        for e in X:
            alpha_prev, beta_prev = omega_x_prev[e]
            alpha, beta = omega_x[e]
            error += pow(alpha_prev-alpha, 2) + pow(beta_prev-beta, 2)
        error = sqrt(error)
        if error < 0.01:
            break
    omega_x = {id_2_edge[e]: alpha_beta for e, alpha_beta in omega_x.items()}
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
def admm(omega, y_t, Y, X, edge_up_nns, edge_down_nns, p0, R, psl = False, approx = False, report_stat = False):
    weight = 1.0
    epsilon = 0.01
    cnt_Y = len(Y)
    cnt_X = len(X)
    cnt_E = cnt_X + cnt_Y
    K = len(R)
    p = calc_initial_p1(y_t, edge_down_nns, X, Y, cnt_E, p0)
    R_p = []
    R_z = [] # R_z is a vector of copied variables in R_p. R_z[e] is a copied variable of R_p[e]
    R_lambda_ = []
    copies = {}
    for k in range(K):
        rule_z = []
        rule_p = []
        rule_lambda_ = []
        for idx, e in enumerate(R[k]):
            rule_z.append(p[e])
            rule_p.append(p[e])
            rule_lambda_.append(0.0)
            if copies.has_key(e):
                copies[e].append([k, idx])
            else:
                copies[e] = [[k, idx]]
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
        p_old = R_p_2_p(R_p, copies, cnt_E)
        if psl == True:
            R_p = psl_update_p(Y, R_p, R_z, R_lambda_, copies, omega, cnt_E, rho)
        else:
            R_p = update_p(X, R_p, R_z, R_lambda_, copies, omega, cnt_E, rho, approx)
        # rho = min(maxRho, rho * 1.1)
        p = R_p_2_p(R_p, copies, cnt_E)
        error = sqrt(np.sum([pow(p_old[e] - p[e], 2) for e in range(cnt_E)]))
        # print ">>>>>>>>>>>> admm iteration.{0}: {1}".format(iter, error)
        if error < epsilon:
            break
    return p


def calc_Omega_from_Obs3(Obs, E,E_X):
    W=2.0
    a=0.5
    T = len(Obs.values()[0])
    Omega = {}
    Opinions={}
    for e in E_X:
        pos_evidence=np.sum(Obs[e])* 1.0
        neg_evidence=T - pos_evidence
        b = pos_evidence/(pos_evidence+neg_evidence+W)
        d = neg_evidence / (pos_evidence + neg_evidence + W)
        u = W/(pos_evidence + neg_evidence + W)
        alpha=W*b/u + W*a
        beta= W*d/u + W*(1-a)
        Opinions[e]=(b,d,u)
        Omega[e] = (alpha,beta)
    return Opinions


def Omega_2_opinion(pred_omega,E_X):
    W=2.0
    a=0.5
    # T = len(Obs.values()[0])
    Omega = {}
    Opinions={}
    for e in E_X.keys():
        r=np.abs(pred_omega[e][0]-W*a)
        s=np.abs(pred_omega[e][1]-W*(1.0-a))
        b=r/(r+s+W)
        d=s/(r+s+W)
        u=W/(r+s+W)
        # b = (pred_omega[0]-1.0)/(pred_omega[0]+pred_omega[1])
        # d = (pred_omega[1]-1.0)/(pred_omega[0]+pred_omega[1])
        # u = W/(pred_omega[0]+pred_omega[1])
        Opinions[e]=(b,d,u)
    return Opinions

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
    if method == 'sl':
        pred_omega_x = SL_prediction_multiCore(V, E, Obs, Omega, copy.deepcopy(E_X))
        pre_opinions = Omega_2_opinion(pred_omega_x, E_X)
    elif method == 'csl':
        pred_omega_x = inference_apdm_format(V, E, Obs, Omega, E_X, logging)
        pre_opinions = Omega_2_opinion(pred_omega_x, E_X)
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
        pre_opinions = Omega_2_opinion(pred_omega_x, E_X)
    elif method == 'csl-conflict-2':
        # b = {e: 0 for e in E}
        b = {}
        psl = False
        pred_omega_x, _ = inference_apdm_format_conflict_evidence(V, E, Obs, Omega, b, E_X, logging, psl)
        pre_opinions = Omega_2_opinion(pred_omega_x, E_X)

    elif method == 'base1':
        pred_omega_x = baseline.base1(V, E, Obs, Omega, E_X)
    elif method == 'base2':
        pred_omega_x = baseline.base2(V, E, Obs, Omega, E_X)
    elif method == 'base3':
        pred_omega_x = baseline.base3(V, E, Obs, Omega, E_X)
    else:
        raise Exception("Method Error")
    alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested = calculate_measures(Omega, pred_omega_x, E_X, logging)
    running_endtime = time.time()
    running_time = running_endtime - running_starttime
    return pre_opinions,alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested, running_time


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
def real_traffic_data_testcase():
    logging = Log()
    data_root = "/network/rit/lab/ceashpc/adil/data/csl-data/Sep18/"   #June25/"
    report_stat = False
    ref_ratios = [0.6, 0.7, 0.8]
    realizations=10
    datasets = ['philly', 'dc']
    count=0
    T = 43
    result = {}
    outf = '../output/test/{}_results-server-traffic-T43-Sep26-opinion.pkl'.format("rr0708")
    methods = ["sl","csl", "csl-conflict-1", "csl-conflict-2", "base1", "base2", "base3"][:4]
    for dataset in datasets[1:]:
        dataroot = data_root + dataset + "/"
        for weekday in range(5)[:1]:
            for hour in range(7, 22)[1:2]:
                for ref_ratio in ref_ratios[1:]: ##########################
                    for test_ratio in [0.1, 0.2, 0.3, 0.4, 0.5][:]:
                        for ratio_conflict in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6][3:5]:
                            for real_i in range(realizations)[:1]:
                                logging.write(str(count)+" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`")
                                count+=1.0
                                for method in methods[:]:
                                    f=dataroot + '/network_{}_weekday_{}_hour_{}_refspeed_{}-testratio-{}-confictratio-{}-realization-{}.pkl'.format(dataset, weekday,hour, ref_ratio, test_ratio,ratio_conflict,real_i)

                                    logging.write("dataset: {} method: {}, weekday:{},hour:{},T, {},ref_ratio:{}, test_ratio: {},conflict_ratio:{}".format(
                                            dataset,method,weekday,hour, T,ref_ratio, test_ratio,ratio_conflict))
                                    logging.write(f)
                                    pkl_file = open(f, 'rb')
                                    [V, E, Obs, E_X, X_b] = pickle.load(pkl_file)
                                    n = len(V)
                                    n_E = len(E)
                                    ndays = len(Obs[E[0]])
                                    T=ndays

                                    Omega = calc_Omega_from_Obs2(Obs, E)
                                    True_Opinions = calc_Omega_from_Obs3(Obs, E, E_X)
                                    pre_Opinions,alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested, running_time = evaluate(V, E, Obs, Omega, E_X, logging, method)
                                    result[
                                        method + "-" + str(test_ratio) + "-" + str(ratio_conflict) + "-" + dataset] = (
                                    True_Opinions, pre_Opinions)
                                    logging.write("prob_mse: {}, running time: {}".format(prob_mse, running_time))
                                    # result_ = {'dataset':dataset,'weekday':weekday,'hour':hour,'ref_ratio':ref_ratio,'network_size': n,
                                    #            'sample_size': ndays - T + 1, 'T': T, 'ratio_conflict': ratio_conflict,
                                    #            'test_ratio': test_ratio, 'acc': (accuracy, accuracy),
                                    #            'prob_mse': (prob_mse, prob_mse),'alpha_mse': (alpha_mse, alpha_mse), 'beta_mse': (beta_mse, beta_mse), 'u_mse': (u_mse, u_mse), 'b_mse': (b_mse, b_mse), 'd_mse': (d_mse, d_mse),'realization':real_i, 'runtime': running_time}
                                    #
    outfp = open(outf, 'w')
    pickle.dump(result, outfp)
    outfp.close()


def main():
    real_traffic_data_testcase()
    # real_traffic_data_testcase_debug()



if __name__=='__main__':
    main()