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
import scipy
# scipy.stats.
# import scipy.stats
from scipy.stats import beta
from cubic import cubic
from log import Log
import time
from basic_funs import *
from network_funs import *

# sys.stdout = open(os.devnull)
# print random.choice(range(10))

# u = random.uniform(0, 1)
# print u, round(u)
# a = [1, 2]
# print np.random.permutation(10)[:5]
# print beta.rvs(1, 1, 0, 1, 100)
# print np.median([0])
# print beta.random_state()
#moduleName = input('cubic')
#import_module(moduleName)
# scipy.stats.

# data=beta.rvs(2,5,loc=0,scale=1,size=5)
# print data
# # print "data", data
# data = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
# data = [0.001, 0.001, 0.00102, 0.00101, 0.001, 0.001, 0.00109, 0.001002, 0.001, 0.001]
# data = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
# data = [0.999, 0.6034349158307533, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999]
# print np.std(data)
# try:
#     alpha, beta, loc, scale = beta.fit(data)
# except:
#     a = 1
# alpha, beta, loc, scale = beta.fit(data, floc=0.,fscale=1.)
# print "alpha: {0}, beta: {1}, loc: {2}, scale: {3}, expected prob: {4}".format(alpha, beta, loc, scale, alpha/(alpha + beta))


# alpha, beta, loc (lower limit), scale (upper limit - lower limit)

"""
INPUT
filename: APDM filename

OUTPUT
V: a list of node ids.
nns: A dictionary with key vertex id, and value the list of neighboring vetex ids
id_2_edge: a dictionary with key a edge id and its value the pair of its two corresponding vertex ids
edge_2_id: a dictionary with key a pair of vertex ids and its value its corresponding edge id
omega: A dictionary of opinions (tuples of alpha and beta values): [edge1 id: [alpha_1, beta_1], ..., edge_M id: [alpha_{cnt_Y+cnt_X}, beta_{cnt_Y+cnt_X}]]. edgei is a pair of vertex ids
feat: A dictionary of lists of observations: [edge1 id: [y_1^1, ..., y_1^T], ..., edgeM id: [y_{cnt_Y+cnt_X}^1, ..., y_{cnt_Y+cnt_X}^T]]
"""
def loadnetwork(filename):
    V = [0, 1, 2]
    nns = {0: [1, 2], 1: [1, 2], 2: [0, 1]}
    id_2_edge = {0: (1,2), 1: (1,2), 2: (0, 2)}
    edge_2_id = {}
    for edge_id, pair in id_2_edge.items():
        edge_2_id[pair] = edge_id
    omega = {0: [0.1, 0.2], 1: [0.3, 0.5], 2: [0.1, 0.6]}
    feat = {0: [1.0, 2.0], 1: [1.5, 1.2], 2: [2.0, 1.1]}
    return V, nns, id_2_edge, edge_2_id, omega, feat


"""
INPUT
id_2_edge: a dictionary with key a edge id and its value the pair of its two corresponding vertex ids

OUTPUT
nns: key is vertex id and value is a list of neighboring vertex ids
"""
def get_nns(edge_id_map):
    nns = {}
    for edge_id, (v1,v2) in edge_id_map.items():
        if nns.has_key(v1):
            if v2 not in nns[v1]:
                nns[v1].append(v2)
        else:
            nns[v1] = [v2]
        if nns.has_key(v2):
            if v1 not in nns[v2]:
                nns[v2].append(v1)
        else:
            nns[v2] = [v1]
    return nns


class Task_inference(object):
    def __init__(self, omega, y_t, Y, X, edge_up_nns, edge_down_nns, p0, R, dict_paths, psl, approx, report_stat):
        self.omega = omega
        self.y_t = y_t
        self.Y = Y
        self.X = X
        self.edge_up_nns = edge_up_nns
        self.edge_down_nns = edge_down_nns
        self.p0 = p0
        self.R = R
        self.dict_paths = dict_paths
        self.psl = psl
        self.approx = approx
        self.report_stat = report_stat

    def __call__(self):
        # this is the place to do your work
        # time.sleep(0.1) # pretend to take some time to do our work
        p_t,sign_grad_py_t = admm(self.omega, self.y_t, self.Y, self.X, self.edge_up_nns,self. edge_down_nns, self.p0, self.R, self.dict_paths, self.psl, self.approx, self.report_stat)
        return p_t,sign_grad_py_t

    def __str__(self):
        return '%s' % (self.p0)


"""
INPUT
Each edge has an ID, an index starting from 0
omega_y: A dictionary of opinions (tuples of alpha and beta values): [edge1 id: [alpha_1, beta_1], ..., edge_M id: [alpha_M, beta_M]]. Suppose the size is cnt_Y.
y: A dictionary of lists of observations: [edge1 id: [y_1^1, ..., y_1^T], ..., edgeM id: [y_M^1, ..., y_M^T]]
Y: The subset of edges that have known opinions
X: The subset of edges whose opinions will be predicted.
edge_nns: key is an edge id, and its value is a list of its neighbor edge ids.
p0: The prior probability that an edge is congested. It is calculated the ratio of total number of congested links indexed by link id and time over the total number of links.
R: a list of neighboring pairs of edges: [(edge1 id, edge2 id), ...]. Each pair has two related PSL rules: edge1 -> edge2 and edge2 -> edge1
init_alpha_beta: the initial pair of alpha and beta for the edges in E_X
psl: True if we want to use pure PSL inference, instead of the inference based on the proposed model; False, otherwise.
approx: True if we want to try the probability values {0.01, 0.02, ..., 0.99} to solve each cubic function equation; False and calculate the exact solution, otherwise.

OUTPUT
omegax: a dictionary with key edge id and value a pair of alpha and beta values
"""
def inference(omega_y, y, Y, X, T, edge_up_nns, edge_down_nns, p0, R, dict_paths, x, logging, psl = False, approx = False, init_alpha_beta = (1, 1), report_stat = False):
    omega = copy.deepcopy(omega_y)
    for e in X.keys(): omega[e] = init_alpha_beta # Beta pdf can be visualized via http://eurekastatistics.com/beta-distribution-pdf-grapher/
    epsilon = 0.01
    maxiter = 1
    error = -1
    for iter in range(maxiter):
        if report_stat:
            logging.write(">>>>>>>>>>>> inference iteration.{0}: {1}".format(iter, error))
        p = []
        sign_grad_py = []

        # Establish communication queues
        tasks = multiprocessing.Queue()
        results = multiprocessing.Queue()

        # Start consumers
        num_consumers = T
        print iter,'Creating %d consumers' % num_consumers
        consumers = [ Consumer(tasks, results)
                      for i in xrange(num_consumers) ]
        for w in consumers:
            w.start()

        num_jobs = T
        # Enqueue jobs
        for t in range(T):
            y_t = {e: e_y[t] for e, e_y in y.items()}
            # x_t = {e: e_x[t] for e, e_x in x.items()}
            tasks.put(Task_inference(omega, y_t, Y, X, edge_up_nns, edge_down_nns, p0, R, dict_paths, psl, approx, report_stat))

        # Add a poison pill for each consumer
        for i in xrange(num_consumers):
            tasks.put(None)

        fin = 0
        # Start printing results
        while num_jobs:
            p_t,sign_grad_py_t = results.get()
            p.append(p_t)
            sign_grad_py.append(sign_grad_py_t)
            num_jobs -= 1

        error = 0.0
        omega_x_prev = {e: alpha_beta for e, alpha_beta in omega.items() if X.has_key(e)}
        # print p
        # print "estimate_omega_x"
        # omega = estimate_omega_x(p, omega, X) #original
        omega = estimate_omega_x(p, X)
        omega_x = {e: alpha_beta for e, alpha_beta in omega.items() if X.has_key(e)}
        for e in X:
            alpha_prev, beta_prev = omega_x_prev[e]
            alpha, beta = omega_x[e]
            error += pow(alpha_prev-alpha, 2) + pow(beta_prev-beta, 2)
        error = sqrt(error)
        if error < 0.01:
            break
    # print omega_x
    return omega_x,sign_grad_py



"""
INPUT
V = [0, 1, ...] is a list of vertex ids
E: a list of pairs of vertex ids
Obs: a dictionary with key edge and its value a list of congestion observations of this edge from t = 1 to t = T
Omega: a dictionary with key edge and its value a pair of alpha and beta.
E_X: Set of edges (pairs of nodes)
init_alpha_beta: the initial pair of alpha and beta for the edges in E_X
psl: True if we want to use pure PSL inference, instead of the inference based on the proposed model; False, otherwise.
approx: True if we want to try the probability values {0.01, 0.02, ..., 0.99} to solve each cubic function equation; False and calculate the exact solution, otherwise.

OUTPUT
omega_x: a dictionary with key edge pair and value a pair of alpha and beta values
"""
def inference_apdm_format(V, E, Obs, Omega, E_X, logging, psl = False, approx = True, init_alpha_beta = (1, 1), report_stat = False):
    # report_stat = True
    if report_stat: print "start reformat"
    edge_up_nns, edge_down_nns, id_2_edge, edge_2_id, omega, feat = reformat(V, E, Obs, Omega)
    omega_y = {}
    p0 = 0.5
    y = {}
    X = {edge_2_id[e]: 1 for e in E_X}
    Y = {e: 1 for e in id_2_edge.keys() if not X.has_key(e)}
    for e in Y.keys():
        omega_y[e] = omega[e]
        y[e] = feat[e]
    T = len(feat[Y.keys()[0]])
    # T = 20
    if report_stat: print "number of time stamps: {}".format(T)
    x = {}
    for e in X.keys():
        x[e] = feat[e]
    if report_stat: print "start generate_PSL_rules_from_edge_cnns"
    R, dict_paths = generate_eopinion_PSL_rules_from_edge_cnns(edge_down_nns, id_2_edge, edge_2_id)
    print "#rules:",len(R)
    # print [id_2_edge[e] for e in R], dict_paths
    # print "id_2_edge", id_2_edge
    # print "dict_paths", dict_paths
    # print "X", X
    # for e in X:
    #     # print "e"
    #     # print e, dict_paths[e]
    #     dict_paths[e]
    if report_stat: print "start inference"
    _,sign_grad_py_ids = inference(omega_y, y, Y, X, T, edge_up_nns, edge_down_nns, p0, R, dict_paths, x, logging, psl, approx, init_alpha_beta, report_stat)
    # omega_x = {id_2_edge[e]: alpha_beta for e, alpha_beta in omega_x.items()}
    # W = 2.0
    # if report_stat:
    #     for e in E_X:
    #         logging.write("************ edge vertex ids: {0}, ".format(e))
    #         logging.write("---- omega_x {0}: {1:.2f}, {2:.2f}".format(Omega[e], omega_x[e][0], omega_x[e][1]))
    #         logging.write("---- uncertainty ({0}): {1}".format(W/(Omega[e][0] + Omega[e][1]), W/(omega_x[e][0] + omega_x[e][1])))
    sign_grad_py = []
    for sign_grad_py_t in sign_grad_py_ids:
        temp_dic = {}
        for id, sign in sign_grad_py_t.items():
            temp_dic[id_2_edge[id]] = sign
        sign_grad_py.append(temp_dic)

    return sign_grad_py




def inference_apdm_format_sliding_window(V, E, Obs, Omega, E_X, begin_time, end_time, window_size, logging, psl = False, approx = True, init_alpha_beta = (1, 1), report_stat = False):
    if report_stat: print "start reformat"
    V, edge_up_nns, edge_down_nns, id_2_edge, edge_2_id, omega, feat = reformat(V, E, Obs, Omega)
    omega_y = {}
    p0 = 0.5
    y = {}
    X = {edge_2_id[e]: 1 for e in E_X}
    Y = {e: 1 for e in id_2_edge.keys() if e not in X}
    for e in Y.keys():
        omega_y[e] = omega[e]
        y[e] = feat[e]
    # print "number of time stamps: {}".format(T)
    # if report_stat: print "number of time stamps: {}".format(T)
    x = {}
    for e in X.keys():
        x[e] = feat[e]
    if report_stat: print "start generate_PSL_rules_from_edge_cnns"
    R = generate_PSL_rules_from_edge_cnns(edge_up_nns)
    if report_stat: print "start inference"
    W = 1.0
    sw_measures = []
    for ws_start in range(begin_time, end_time+1):
        print "sliding window: {0} to {1}".format(ws_start, ws_start + window_size)
        sw_omega_x, sw_omega_y, sw_x, sw_y = sliding_window_extract(x, y, ws_start, window_size)
        pred_omega_x = inference(sw_omega_y, sw_y, Y, X, window_size, edge_up_nns, edge_down_nns, p0, R, sw_x, logging, psl, approx, init_alpha_beta, report_stat)
        # pred_omega_x = {id_2_edge[e]: alpha_beta for e, alpha_beta in pred_omega_x.items()}
        prob_mse, u_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested = calculate_measures(sw_omega_x, W, pred_omega_x, X, logging)
        sw_measures.append([prob_mse, u_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested])
    avg_measures = [0, 0, 0, 0, 0, 0, 0]
    for measures in sw_measures:
        avg_measures = list_sum(avg_measures, measures)
    avg_measures = list_dot_scalar(avg_measures, 1.0 / len(sw_measures))
    logging.write("prob_mse: {0}, u_mse: {1}, prob_relative_mse: {2}, u_relative_mse: {3}".format(prob_mse, u_mse, prob_relative_mse, u_relative_mse))
    logging.write("accuracy: {0}, recall_congested: {1}, recall_uncongested: {2}".format(accuracy, recall_congested, recall_uncongested))
    return sw_measures, avg_measures


# def evaluation():
#     sw_measures = []
#     for ws_start in range(begin_time, end_time+1):
#         print "sliding window: {0} to {1}".format(ws_start, ws_start + window_size)
#         sw_omega_x, sw_omega_y, sw_x, sw_y = sliding_window_extract(x, y, ws_start, window_size)
#         pred_omega_x = inference(sw_omega_y, sw_y, Y, X, window_size, edge_up_nns, edge_down_nns, p0, R, sw_x, logging, psl, approx, init_alpha_beta, report_stat)
#         # pred_omega_x = {id_2_edge[e]: alpha_beta for e, alpha_beta in pred_omega_x.items()}
#         prob_mse, u_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested = calculate_measures(sw_omega_x, W, pred_omega_x, X, logging)
#         sw_measures.append([prob_mse, u_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested])
#     avg_measures = [0, 0, 0, 0, 0, 0, 0]
#     for measures in sw_measures:
#         avg_measures = list_sum(avg_measures, measures)
#     avg_measures = list_dot_scalar(avg_measures, 1.0 / len(sw_measures))
#     logging.write("prob_mse: {0}, u_mse: {1}, prob_relative_mse: {2}, u_relative_mse: {3}".format(prob_mse, u_mse, prob_relative_mse, u_relative_mse))
#     logging.write("accuracy: {0}, recall_congested: {1}, recall_uncongested: {2}".format(accuracy, recall_congested, recall_uncongested))


"""
INPUT
Omega: key is edge pair, and value is (alpha, beta)
Obs: key is edge pair, and value is a list of observations
start_t: the start time stamp that the sliding window should start from
window_size: the size of the sliding window

OUTPUT
sw_Omega: Same format as Omega, but with opinions updated based on the observations of the current time window
sw_Obs: Same format as Obs, but with observations within the time window
"""
def sliding_window_extract(Omega, Obs, start_t, window_size):
    sw_Omega = {}
    sw_Obs = {}
    for e, obs in Obs.items():
        sw_Obs[e] = [Obs[e][t] for t in range(start_t, start_t+window_size)]
        n = np.sum(sw_Obs[e])
        sw_Omega[e] = (n + 0.001, window_size - n + 0.001)
    return sw_Omega, sw_Obs

#
# def sliding_window_extract(x, y, start_t, window_size):
#     sw_omega_x = {}
#     sw_omega_y = {}
#     sw_x = {}
#     sw_y = {}
#     for e, obs in y.items():
#         sw_y[e] = [y[e][t] for t in range(start_t, start_t+window_size)]
#         n_y = np.sum(sw_y[e])
#         sw_omega_y[e] = (n_y, window_size - n_y)
#     for e, obs in x.items():
#         sw_x[e] = [x[e][t] for t in range(start_t, start_t+window_size)]
#         n_x = np.sum(sw_x[e])
#         sw_omega_x[e] = (n_x, window_size - n_x)
#     return sw_omega_x, sw_omega_y, sw_x, sw_y

"""
INPUT
p: [p_1, p_2, ... p_T]. p_t is a list of truth probability values of all edges. p[t][i] refers to the probability value of edge id=i at time t, where t \in {1, ..., T} and i \in {1, cnt_Y+cnt_X}
omega: A dictionary of opinions (tuples of alpha and beta values) of all edges (X+Y): [edge1 id: [alpha_1, beta_1], ...]. Note that the opinions related to the edges in X will be udpated based on p.
X: a list of edge ids whose opinions need to be predicted

OUTPUT
omega_x: A dictionary of opinions (tuples of alpha and beta values): [edge1 id: [alpha_1, beta_1], ..., edge_N id: [alpha_M, beta_N]]. cnt_X is the size of X.
"""
def estimate_omega_x1(p, omega, X):
    # for e in X:
        # data = [p_t[e] for p_t in p]
        # alpha = sum(data)
        # beta = len(data) -
    for e in X:
        data = [p_t[e] for p_t in p]
        if np.std(data) < 0.01:
            alpha1 = np.mean(data)
            beta1 = 1 - alpha1
        else:
            data = [max([p_t[e] - random.random() * 0.01,0]) for p_t in p]
            alpha1, beta1, loc, scale = beta.fit(data, floc=0.,fscale=1.)
            if alpha1<1:
                beta1 = 1.1 * beta1 / alpha1
                alpha1 = 1.1
            if beta1 + alpha1 > 10:
                alpha1 = alpha1 / (alpha1 + beta1) * 10
                beta1 = 10 - alpha1
            # print alpha1, beta1
            # print alpha1, beta1
        omega[e] = (alpha1, beta1)
    return omega

#
# def estimate_omega_x(p, omega, X):
#     for e in X:
#         data = [p_t[e] for p_t in p]
#         if np.std(data) < 0.01:
#             alpha1 = np.mean(data)
#             beta1 = 1 - alpha1
#         else:
#             alpha1, beta1, loc, scale = beta.fit(data, floc=0,fscale=1)
#         # alpha1 = alpha1 / (alpha1 + beta1) * 10
#         # beta1 = 10 - alpha1
#         omega[e] = (alpha1, beta1)
#     return omega


def list_dot_scalar(l, c):
    return [item * c for item in l]

def list_sum(l1, l2):
    l = []
    for v1, v2 in zip(l1, l2):
        l.append(v1 + v2)
    return l

def list_minus(l1, l2):
    # print "l1, l2", l1, l2
    l = []
    for v1, v2 in zip(l1, l2):
        l.append(v1 - v2)
    return l

def list_dot(l1, l2):
    l = []
    for v1, v2 in zip(l1, l2):
        l.append(v1 * v2)
    return l


def get_edge_id(v1, v2, edge2id):
    if v1 <= v2:
        return edge2id[(v1, v2)]
    else:
        return edge2id[(v2, v1)]




"""
INPUT
nns: key is vertex id and value is a list of neighboring vertex ids
edge2id: key is edge and value is the id of this edge. Each edge is represented by a pair of vertices with order.

OUTPUT
R: a list of pairs of neighboring edge ids.
"""
def generate_PSL_rules_from_nns(nns, edge_2_id):
    dic_R = {}
    for v, neighbors in enumerate(nns):
        for v_n in neighbors:
            e1 = get_edge_id(v, v_n, edge_2_id)
            for v_nn in nns[v_n]:
                if v_nn != v:
                    e2 = get_edge_id(v_n, v_nn, edge_2_id)
                    if e1 < e2:
                        if not dic_R.has_key((e1, e2)):
                            dic_R[(e1, e2)] = 1
                    else:
                        if not dic_R.has_key((e2, e1)):
                            dic_R[(e2, e1)] = 1
    return dic_R.keys()






def generate_PSL_rules_from_edge_cnns1(edge_up_nns):
    dic_R = {}
    for e, neighbors in edge_up_nns.items():
        for up_e in neighbors.keys():
            if not dic_R.has_key((e, up_e)):  # The direction should be from up_e to e. The traffic at e will impact the traffic of its up adjacent edges
                dic_R[(e, up_e)] = 1
    return dic_R.keys()


"""
INPUT
x, y: two dictionaries

OUPUT
z: The merged dictionary
"""
def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

class Task_P(object):
    def __init__(self, e, omega_e, dict_R_z, dict_R_lambda_, rho, copies_e, aa, bb, cc, dd, approx):
        self.e = e
        self.omega_e = omega_e
        self.dict_R_z = dict_R_z
        self.dict_R_lambda_ = dict_R_lambda_
        self.rho = rho
        self.copies_e = copies_e
        self.aa = aa
        self.bb = bb
        self.cc = cc
        self.dd = dd
        self.approx = approx

    def __call__(self):
        # this is the place to do your work
        if self.approx:
            min_prob = sol_min_p_approx(self.e, self.omega_e, self.dict_R_z, self.dict_R_lambda_, self.rho, self.copies_e, self.aa, self.bb, self.cc, self.dd)
        else:
            min_prob = sol_min_p(self.e, self.omega_e, self.dict_R_z, self.dict_R_lambda_, self.rho, self.copies_e, self.aa, self.bb, self.cc, self.dd)
        return self.e, min_prob

    def __str__(self):
        return '%s' % (self.rho)


"""
INPUT
R_p: R_p[k][j] refers to the probability variable of j-th edge in the k-th rule in R.
R_z: R_p[k][j] refers to the probability copy variable of j-th edge in the k-th rule in R.
R_lambda_: R_lambda_[k][j] refers to the augmented lagrange multiplier variable related to the j-th edge in the k-th rule in R.
copies: copies[e] is a list indexes of the variables in the set of rules R that are related to the edge e.
        for k, j in copies[e]:
            k is the index of a specific rule and j is index of a specific edge in the rule.
omega: A dictionary of opinions (tuples of alpha and beta values) of all edges (X+Y): [edge1 id: [alpha_1, beta_1], ...]. The opinions of edges in X are the current estimates.
cnt_Y: size of Y
cnt_X: size of X
rho: step size.

OUTPUT
R_p: The probability values in R_p are updated based on R_z and R_lambda_.

"""
def update_p(X, R_p, R_z, R_lambda_, copies, omega, cnt_E, rho, approx):
    for e in X.keys():
        """
        -rho |copies(p_i)| p_i^3 + (rho|copies(p_i)| + rho \sum_{z[k][j]\in copies(p_i)} (z[k][j] + lambda[k][j]/rho)) p_i^2 +
        (omega[i][0] + omega[i][1] - 2 - rho \sum_{z[k][j]\in copies(p_i)} (z[k][j] + lambda[k][j]/rho))p_i + 1 - omega[i][0] = 0

        Let a = -2 |copies(p_i)|
        b = (rho|copies(p_i)| + rho \sum_{z[k][j]\in copies(p_i)} (z[k][j] + lambda[k][j]/rho))
        c = (omega[i][0] + omega[i][1] - 2 - rho \sum_{z[k][j]\in copies(p_i)} (z[k][j] + lambda[k][j]/rho))
        d = 1 - omega[i][0]
        We obtain the following cubic euqation:
        a * p_i^3 + b * p_i^2 + c * p_i + d = 0
        """
        if not copies.has_key(e): continue
        nc = len(copies[e])
        aa = -1 * rho * nc
        z_lambda_sum = sum([R_z[k][j] + R_lambda_[k][j] / rho for k, j in copies[e]])
        bb = rho * nc + rho * z_lambda_sum
        cc = omega[e][0] + omega[e][1] - 2 - rho * z_lambda_sum
        dd = 1 - omega[e][0]
        if approx:
            min_prob = sol_min_p_approx(e, omega, R_z, R_lambda_, rho, copies, aa, bb, cc, dd)
        else:
            min_prob = sol_min_p(e, omega, R_z, R_lambda_, rho, copies, aa, bb, cc, dd)
        for k, j in copies[e]:
            R_p[k][j] = min_prob
    return R_p


"""
-rho |copies(p_i)| p_i^3 + (rho|copies(p_i)| + rho \sum_{z[k][j]\in copies(p_i)} (z[k][j] + lambda[k][j]/rho)) p_i^2 +
(omega[i][0] + omega[i][1] - 2 - rho \sum_{z[k][j]\in copies(p_i)} (z[k][j] + lambda[k][j]/rho))p_i + 1 - omega[i][0] = 0

Let a = -2 |copies(p_i)|
b = (rho|copies(p_i)| + rho \sum_{z[k][j]\in copies(p_i)} (z[k][j] + lambda[k][j]/rho))
c = (omega[i][0] + omega[i][1] - 2 - rho \sum_{z[k][j]\in copies(p_i)} (z[k][j] + lambda[k][j]/rho))
d = 1 - omega[i][0]
We obtain the following cubic euqation:
a * p_i^3 + b * p_i^2 + c * p_i + d = 0
"""
#                    sign_grad_py_t, p, Y, R_z, R_lambda_, copies, omega, rho
def get_sign_grad_py(sign_grad_py_t, p, Y, R_z, R_lambda_, copies, omega,rho):
    for e in Y.keys():
        # if not copies.has_key(e): continue
        if not copies.has_key(e):
            if sign_grad_py_t.has_key(e):
                sign_grad_py_t[e].append(0.0)  #the edges  are not involved any rules
            else:
                sign_grad_py_t[e]=[0.0]
            continue
        z_lambda_sum = sum([p[e]-R_z[k][j] - R_lambda_[k][j] / rho for k, j in copies[e]])
        term1 =-(omega[e][0]-1)*(1-p[e]) +(omega[e][1]-1)*p[e]
        term2= rho*(1-p[e])*p[e]*z_lambda_sum

        grad = term1+term2
        if grad >= 0:
            if sign_grad_py_t.has_key(e):
                sign_grad_py_t[e].append(1.0)
            else:
                sign_grad_py_t[e] = [1.0]
        else:
            if sign_grad_py_t.has_key(e):
                sign_grad_py_t[e].append(-1.0)
            else:
                sign_grad_py_t[e] = [-1.0]
    return sign_grad_py_t


"""
This function returns the solution to
a * p_i^3 + b * p_i^2 + c * p_i + d = 0
that at the same time minimizes the objective function

"""
def sol_min_p1(e, omega_e, dict_R_z, dict_R_lambda_, rho, copies_e, aa, bb, cc, dd):
    probs = cubic(aa, bb, cc, dd)
    probs = [prob for prob in probs if type(prob) is not complex and prob > 0 and prob < 1]
    probs.extend([0.001, 0.999])
    min_prob = -1
    min_score = float('inf')
    for prob in probs:
        score = -1 * (omega_e[0] - 1) * log(prob) - (omega_e[1] - 1) * log(1-prob) + rho * 0.5 * sum([pow(prob - dict_R_z[k][j] - dict_R_lambda_[k][j] / rho, 2) for k, j in copies_e])
        if score < min_score:
            min_prob = prob
            min_score = score
    if min_prob == -1:
        debug = 1
    return min_prob


def sol_min_p(e, omega, R_z, R_lambda_, rho, copies, aa, bb, cc, dd):
    probs = cubic(aa, bb, cc, dd)
    probs = [prob for prob in probs if type(prob) is not complex and prob > 0 and prob < 1]
    probs.extend([0.001, 0.999])
    min_prob = -1
    min_score = float('inf')
    for prob in probs:
        score = -1 * (omega[e][0] - 1) * log(prob) - (omega[e][1] - 1) * log(1-prob) + rho * 0.5 * sum([pow(prob - R_z[k][j] - R_lambda_[k][j] / rho, 2) for k, j in copies[e]])
        if score < min_score:
            min_prob = prob
            min_score = score
    if min_prob == -1:
        debug = 1
    return min_prob

def sol_min_p_approx1(e, omega_e, dict_R_z, dict_R_lambda_, rho, copies_e, aa, bb, cc, dd):
    min_prob = -1
    min_score = float('inf')
    for prob in np.arange(0.01, 1, 0.01):
        score = -1 * (omega_e[0] - 1) * log(prob) - (omega_e[1] - 1) * log(1-prob) + rho * 0.5 * sum([pow(prob - dict_R_z[k][j] - dict_R_lambda_[k][j] / rho, 2) for [k, j] in copies_e])
        if score < min_score:
            min_prob = prob
            min_score = score
    if min_prob == -1:
        debug = 1
    return min_prob

def sol_min_p_approx(e, omega, R_z, R_lambda_, rho, copies, aa, bb, cc, dd):
    min_prob = -1
    min_score = float('inf')
    probs=list(np.arange(0.1, 1, 0.1))
    probs.extend([0.001,0.01,0.99,0.999])
    for prob in probs:
        score = -1 * (omega[e][0] - 1) * log(prob) - (omega[e][1] - 1) * log(1-prob) + rho * 0.5 * sum([pow(prob - R_z[k][j] - R_lambda_[k][j] / rho, 2) for k, j in copies[e]])
        if score < min_score:
            min_prob = prob
            min_score = score
    if min_prob == -1:
        debug = 1
    return min_prob

def psl_update_p(Y, R_p, R_z, R_lambda_, copies, omega, cnt_E, rho):
    for e in range(cnt_E):
        if Y.has_key(e): continue
        nc = len(copies[e]) * 1.0
        z_lambda_sum = sum([R_z[k][j] + R_lambda_[k][j] / rho for k, j in copies[e]])
        prob = z_lambda_sum / nc
        if prob < 0: prob = 0
        if prob > 1: prob = 1
        for k, j in copies[e]:
            R_p[k][j] = prob
    return R_p



"""
INPUT
y_t: A dictionary of lists of observations at time t with key an edge id and value its observation at time t.
Y: The subset of edge ids that have known opinions
edge_nns: key is an edge id, and its value is a list of its neighbor edge ids.
cnt_E: The total number of edges
cnt_X: the number of edges whose opinions will be predicted (size of X).
p0: The prior probability that an edge is congested. It is calculated the ratio of total number of congested links indexed by link id and time over the total number of links.
OUTPUT
p: a list of probability initial values of the cnt_Y + cnt_X edges. p[i] refers to the initial probability value of edge id=i
"""
def calc_initial_p1(dict_paths, y_t, edge_down_nns, X, Y, cnt_E, p0):
    p = [0 for i in range(cnt_E)]
    for e in Y:
        p[e] = y_t[e]
    for e in X:
        conf = 0
        n_post = 0
        if e in dict_paths:
            for (e1, e2) in dict_paths[e]:
                conf += p[e1] * p[e2]
                n_post += p[e1] + p[e2]
            if conf > 0 or n_post > 1:
                p[e] = 1.0
            else:
                p[e] = 0.0
        else:
            p[e] = 0



    for e in X:
        conf = 0.0
        n_post = 0.0
        n_zero=0.0
        if e in dict_paths:
            for (e1, e2) in dict_paths[e]:
                if p[e1] * p[e2] > 0:
                    conf += p[e1] * p[e2]

                if p[e1] + p[e2]>0:
                    n_post +=1.0
                if p[e1] + p[e2]==0:
                    n_zero +=1.0

            if conf >=n_zero :
                p[e] = 1.0
            else:
                if n_post>=conf:
                    p[e] = 1.0
                else:
                    p[e] = 0.0
        else:
            p[e] = 0

    return p

#0.6....
def calc_initial_p(dict_paths, y_t, edge_down_nns, X, Y, cnt_E, p0):
    p = [0 for i in range(cnt_E)]
    for e in Y:
        p[e] = y_t[e]

    for e in X:
        conf = 0
        n_post = 0
        if e in dict_paths:
            for (e1, e2) in dict_paths[e]:
                conf += p[e1] * p[e2]
                n_post += p[e1] + p[e2]
            if conf > 0:
                p[e] = 1
            else:
                if n_post >1.0:
                    p[e] = 1.0
                else:
                    p[e] = 0.0
        else:
            p[e] = 0

    return p

#0.2
def calc_initial_p2(dict_paths, y_t, edge_down_nns, X, Y, cnt_E, p0):
    p = [0 for i in range(cnt_E)]
    for e in Y:
        p[e] = y_t[e]
    # print "dict", dict_paths
    for i in range(3):
        for e in X:
            conf = 0
            n_post = 0
            if e in dict_paths:
                for (e1, e2) in dict_paths[e]:
                    conf += p[e1] * p[e2]
                    n_post += p[e1] + p[e2]
                if conf > 0:
                    p[e] = 1
            else:
                p[e] = 0

    for e in X:
        conf = 0
        n_post = 0
        if e in dict_paths:
            for (e1, e2) in dict_paths[e]:
                conf += p[e1] * p[e2]
                n_post += p[e1] + p[e2]
            if conf > 0:
                p[e] = 1
            else:
                if n_post >1.0:
                    p[e] = 1.0
                else:
                    p[e] = 0.0
        else:
            p[e] = 0

    return p

#0.2
def calc_initial_p2_2(dict_paths, y_t, edge_down_nns, X, Y, cnt_E, p0):
    p = [0 for i in range(cnt_E)]
    for e in Y:
        p[e] = y_t[e]
    # print "dict", dict_paths
    for i in range(3):
        for e in X:
            conf = 0
            n_post = 0
            if e in dict_paths:
                for (e1, e2) in dict_paths[e]:
                    conf += p[e1] * p[e2]
                    n_post += p[e1] + p[e2]
                if conf > 0:
                    p[e] = 1
            else:
                p[e] = 0

    for e in X:
        conf = 0
        n_post = 0
        if e in dict_paths:
            for (e1, e2) in dict_paths[e]:
                conf += p[e1] * p[e2]
                n_post += p[e1] + p[e2]
            if conf > 0:
                p[e] = 1
            else:
                if n_post >0.0:
                    p[e] = 1.0
                else:
                    p[e] = 0.0
        else:
            p[e] = 0

    return p
# def calc_initial_p(dict_paths, y_t, edge_down_nns, X, Y, cnt_E, p0):
#     p = [0 for i in range(cnt_E)]
#     for e in Y:
#         p[e] = y_t[e]
#     for e in X:
#         conf = 0
#         n_post = 0
#         for (e1,e2) in dict_paths[e]:
#             conf += p[e1] * p[e2]
#             n_post += p[e1] + p[e2]
#         if conf > 0:
#             p[e] = 1
#         else:
#             if n_post == 0:
#                 p[e] = 0.0
#             else:
#                 p[e] = 0.0
#     return p


"""
INPUT
v: a vector of numbers that may be outside the range [0, 1]

OUTPUT
v: truncated vector of numbers in the range [0, 1]
"""
def normalize(v):
    for i in range(len(v)):
        if v[i] < 0: v[i] = 0
        if v[i] > 1: v[i] = 1
    return v

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




"""
project v to the following hyperplane
v[0] + v[1] - v[2] - 1 = 0

v[0] + a
v[1] + b
v[2] + c

v[0] + v[1] - v[2] - 1 + a + b - c = 0

min_{a,b,c \in [0, 1]} a^2 + b^2 + c^2

is equivalent to

min_{a,b,c \in [0, 1]}  a^2 + b^2 + (v[0] + v[1] - v[2] - 1 + a + b)^2

let const = v[0] + v[1] - v[2] - 1. The objective function becomes

a^2 + b^2 + (const + a + b)^2

Let x = [a, b]. The objective function becomes

1^T x + (const + 1^T x)^2

The gradient is:

1 + const * 1 + 1^Tx 1 = 0 ==> x[0] = x[1] = d

d + const + 2d = 0 ==> d = - const / 3

a = b = const / 3, c = v[0] + v[1] - v[2] - 1 + a + b = const + a + b

"""
def projection(v):
    const = v[0] + v[1] - v[2] - 1 # negative value
    a = -1 * const / 3.0
    b = -1 * const / 3.0
    c = const + a + b
    return list_sum(v, [a,b,c])

def ell_fun(v):
    return v[0] + v[1] - v[2] - 1


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
def admm(omega, y_t, Y, X, edge_up_nns, edge_down_nns, p0, R, dict_paths, psl = False, approx = False, report_stat = False):
    weight = 1.0
    epsilon = 0.01
    cnt_Y = len(Y)
    cnt_X = len(X)
    cnt_E = cnt_X + cnt_Y
    K = len(R)
    y_prob = {}
    sign_grad_py_t = {}
    # print "omega", omega
    # print "y_t", y_t
    # for e in y_t.keys():
    #     y_prob[e] =  beta.rvs(omega[e][0],omega[e][1],loc=0.,scale=1.,size=1)[0]
    p = calc_initial_p2_2(dict_paths, y_t, edge_down_nns, X, Y, cnt_E, p0)
    # for e in y_t.keys():
    #     y_prob[e] =  beta.rvs(omega[e][0],omega[e][1],loc=0.,scale=1.,size=1)[0]
    # p = calc_initial_p(dict_paths, y_prob, edge_down_nns, X, Y, cnt_E, p0)
    R_p = []
    R_z = [] # R_z is a vector of copied variables in R_p. R_z[e] is a copied variable of R_p[e]
    R_lambda_ = []
    copies = {}
    for k in xrange(K):
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
    rho = 1
    maxiter = 30
    # maxRho = 5
    for iter in range(maxiter):
        for k in xrange(K):
            # update lambda variables
            R_lambda_[k] = list_sum(R_lambda_[k], list_dot_scalar(list_minus(R_z[k], R_p[k]), rho))
            # update copy variables
            R_z[k] = normalize(list_minus(R_p[k], list_dot_scalar(R_lambda_[k], -1.0 / rho)))
            if ell_fun(R_z[k]) > 0:
                """
                The minimization problem to be solved is:
                    min weight * [1, 1, -1] * zxy[k] - 1 + rho/2 || zxy[k] - pxy[k] + 1/rho * lambda_xy[k] ||_2^2

                The gradient of the objective function over zxy[k] is:
                    weight * [1, 1, -1] + rho (zxy[k] - pxy[k] + 1/rho * lambda_xy[k])

                The updates zxy[k] can be identified such that the above gradient equals to 0:
                    zxy[k] = pxy[k] - 1/rho * lambda_xy[k] - weight * 1/rho [1, 1, -1]
                """
                R_z[k] = normalize(list_minus(R_p[k], list_sum(list_dot_scalar(R_lambda_[k], 1/rho), [weight/rho, weight/rho, -1 * weight/rho])))
                if ell_fun(R_z[k]) < 0:
                    R_z[k] = normalize(projection(list_minus(R_p[k], list_dot_scalar(R_lambda_[k], 1.0/rho))))
        # update probability variables
        p_old = R_p_2_p(R_p, copies, cnt_E)
        if psl == True:
            R_p = psl_update_p(Y, R_p, R_z, R_lambda_, copies, omega, cnt_E, rho)
        else:
            R_p = update_p(X, R_p, R_z, R_lambda_, copies, omega, cnt_E, rho, approx)
        # rho = min(maxRho, rho * 1.1)
        p = R_p_2_p(R_p, copies, cnt_E)

        sign_grad_py_t = get_sign_grad_py(sign_grad_py_t, p, Y, R_z, R_lambda_, copies, omega, rho)
        error = sqrt(np.sum([pow(p_old[e] - p[e], 2) for e in range(cnt_E)]))
        # print ">>>>>>>>>>>> admm iteration.{0}: {1}".format(iter, error)
        if error < epsilon:
            break
    return p,sign_grad_py_t



"""
INPUT
R_p: R_p[k][j] refers to the probability variable of j-th edge in the k-th rule in R.
copies: copies[e] is a list indexes of the variables in the set of rules R that are related to the edge e.
        for k, j in copies[e]:
            k is the index of a specific rule and j is index of a specific edge in the rule.
cnt_E: total number of edges in the network

OUTPUT
p: a list of porbability values. p[i] is the probability value of the edge i.
"""
def R_p_2_p(R_p, copies, cnt_E):
    p = [-1 for i in range(cnt_E)]
    for e, e_copies in copies.items():
        k, j = e_copies[0]
        p[e] = R_p[k][j]
    return p



# """
# INPUT
# V = [0, 1, ...] is a list of vertex ids
# E: a list of pairs of vertex ids
# Obs: a dictionary with key edge and its value a list of congestion observations of this edge from t = 1 to t = T
# Omega: a dictionary with key edge and its value a pair of alpha and beta
#
# OUTPUT
# V: same as input
# edge_up_nns: key is an edge id, and its value is a list of its up stream neighbor edge ids.
# edge_down_nns: key is an edge id, and its value is a list of its down stream neighbor edge ids.
# edge_2_id: a dictionary with key a directed edge and its value is the id of this edge.
# id_2_edge: a dictionary with key an edge id and its value the corresponding edge
# omega: A dictionary of opinions (tuples of alpha and beta values): [edge1 id: [alpha_1, beta_1], ..., ]. The size of the dictionary is the total number of edges.
# feat: A dictionary of lists of observations: [edge1 id: [y_1^1, ..., y_1^T], ...].
# """
# def reformat(V, E, Obs, Omega):
#     feat = {}
#     omega = {}
#     id_2_edge = {}
#     edge_2_id = {}
#     for idx, e in enumerate(E):
#         id_2_edge[idx] = e
#         edge_2_id[e] = idx
#     for e, e_feat in Obs.items():
#         feat[edge_2_id[e]] = e_feat
#     for e, alpha_beta in Omega.items():
#         omega[edge_2_id[e]] = alpha_beta
#     edges_end_at = {}
#     edges_start_at = {}
#     for e in E:
#         e_id = edge_2_id[e]
#         start_v = e[0]
#         end_v = e[1]
#         if edges_end_at.has_key(end_v):
#             edges_end_at[end_v][e_id] = 1
#         else:
#             edges_end_at[end_v] = {e_id: 1}
#         if edges_start_at.has_key(start_v):
#             edges_start_at[start_v][e_id] = 1
#         else:
#             edges_start_at[start_v] = {e_id: 1}
#     edge_up_nns = {}
#     edge_down_nns = {}
#     for e in E:
#         start_v = e[0]
#         end_v = e[1]
#         e_id = edge_2_id[e]
#         if edges_end_at.has_key(start_v):
#             edge_up_nns[e_id] = edges_end_at[start_v]
#         if edges_start_at.has_key(end_v):
#             edge_down_nns[e_id] = edges_start_at[end_v]
#     V = {v: 1 for v in V}
#     return V, edge_up_nns, edge_down_nns, id_2_edge, edge_2_id, omega, feat

def testcase6():
    logging = Log('Testcase6.txt')
    V = [0,1,2,3,4]
    E = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 3), (2, 1), (3, 4)]
    Obs = {(0, 1): [1], (0, 2): [1], (0, 3): [0], (0, 4): [0], (1, 3): [0], (2, 1): [1], (3, 4): [1]}
    E_X = [(0, 1), (0, 3), (0, 4)]
    logging.write("\r\n\r\ntest case 6: ##########################")
    Omega = {}
    for e in E:
        if np.mean(Obs[e]) > 0.5:
            Omega[e] = (10, 1)
        else:
            Omega[e] = (1, 10)
    evaluate(V, E, Obs, Omega, E_X, logging, psl = False, approx = False, init_alpha_beta = (1, 1), report_stat = False)


    V = [0,1,2,3,4]
    E = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 3), (2, 1), (3, 4)]
    Obs = {(0, 1): [1], (0, 2): [1], (0, 3): [1], (0, 4): [1], (1, 3): [1], (2, 1): [1], (3, 4): [1]}
    E_X = [(0, 1), (0, 3), (0, 4)]
    logging.write("\r\n\r\ntest case 6: ##########################")
    Omega = {}
    for e in E:
        if np.mean(Obs[e]) > 0.5:
            Omega[e] = (10, 1)
        else:
            Omega[e] = (1, 10)
    evaluate(V, E, Obs, Omega, E_X, logging, psl = False, approx = False, init_alpha_beta = (1, 1), report_stat = False)


    # V = [0,1,2,3,4]
    # E = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 3), (2, 1), (3, 4)]
    # Obs = {(0, 1): [1], (0, 2): [1], (0, 3): [1], (0, 4): [1], (1, 3): [1], (2, 1): [1], (3, 4): [1]}
    # E_X = [(0, 1), (0, 3), (0, 4)]
    # logging.write("\r\n\r\ntest case 6: ##########################")
    # Omega = {}
    # for e in E:
    #     if np.mean(Obs[e]) > 0.5:
    #         Omega[e] = (10, 1)
    #     else:
    #         Omega[e] = (1, 10)
    # evaluate(V, E, Obs, Omega, E_X, logging, psl = False, approx = False, init_alpha_beta = (1, 1), report_stat = False)
    #

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
    evaluate(V, E, Obs, Omega, E_X, logging, psl = False, approx = False, init_alpha_beta = (1, 1), report_stat = False)

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
    evaluate(V, E, Obs, Omega, E_X, logging, psl = False, approx = False, init_alpha_beta = (1, 1), report_stat = False)


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
    evaluate(V, E, Obs, Omega, E_X, logging, psl = False, approx = False, init_alpha_beta = (1, 1), report_stat = False)


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
    logging.write("The edges 7->5 and 7->2 should be congested")
    logging.write("init_alpha_beta = (1, 1), which means equal chance to be congested or uncongested!\r\n ")
    Omega = {}
    for e in E:
        if np.mean(Obs[e]) > 0.5:
            Omega[e] = (10,1)
        else:
            Omega[e] = (1, 10)
    evaluate(V, E, Obs, Omega, E_X, logging, psl = False, approx = False, init_alpha_beta = (1, 1), report_stat = False)


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
    logging.write("The edges 7->5, 7->2, and 4->2 should be congested")
    logging.write("init_alpha_beta = (1, 1), which means equal chance to be congested or uncongested!\r\n ")
    Omega = {}
    for e in E:
        if np.mean(Obs[e]) > 0.5:
            Omega[e] = (10,1)
        else:
            Omega[e] = (1, 10)
    evaluate(V, E, Obs, Omega, E_X, logging, psl = False, approx = False, init_alpha_beta = (1, 1), report_stat = False)



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
    logging.write("The edges 7->2 and 4->2 should be congested")
    logging.write("init_alpha_beta = (1, 1), which means equal chance to be congested or uncongested!\r\n ")
    Omega = {}
    for e in E:
        if np.mean(Obs[e]) > 0.5:
            Omega[e] = (10,1)
        else:
            Omega[e] = (1, 10)
    evaluate(V, E, Obs, Omega, E_X, logging, psl = False, approx = False, init_alpha_beta = (1, 1), report_stat = False)

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
    logging.write("The edges 7->5, 4->2, and 7->2 should be congested")
    logging.write("init_alpha_beta = (1, 1), which means equal chance to be congested or uncongested!\r\n ")
    Omega = {}
    for e in E:
        if np.mean(Obs[e]) > 0.5:
            Omega[e] = (10,1)
        else:
            Omega[e] = (1, 10)
    evaluate(V, E, Obs, Omega, E_X, logging, psl = False, approx = False, init_alpha_beta = (1, 1), report_stat = False)


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
    logging.write("The edges 7->5, 4->2, and 7->2 should be congested")
    logging.write("init_alpha_beta = (1, 1), which means equal chance to be congested or uncongested!\r\n ")
    Omega = {}
    for e in E:
        if np.mean(Obs[e]) > 0.5:
            Omega[e] = (10,1)
        else:
            Omega[e] = (1, 10)
    evaluate(V, E, Obs, Omega, E_X, logging, psl = False, approx = False, init_alpha_beta = (1, 1), report_stat = False)

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
    logging.write("The edges 7->5, 4->2, 7->2, and 3->1 should be congested")
    logging.write("init_alpha_beta = (1, 1), which means equal chance to be congested or uncongested!\r\n ")
    Omega = {}
    for e in E:
        if np.mean(Obs[e]) > 0.5:
            Omega[e] = (10,1)
        else:
            Omega[e] = (1, 10)
    evaluate(V, E, Obs, Omega, E_X, logging, psl = False, approx = False, init_alpha_beta = (1, 1), report_stat = False)



    # Obs[(7,8)] = [1]
    # Obs[(8,5)] = [1]
    # Obs[(5,2)] = [0]
    # Obs[(4,2)] = [0]
    # Obs[(7,2)] = [0]

    # Obs[(7,4)] = [1]
    # Obs[(4,1)] = [1]
    # Obs[(1,2)] = [1]
    # Obs[(4,2)] = [0]
    # Obs[(7,2)] = [0]


    # Obs[(6,3)] = [1]
    # Obs[(3,0)] = [1]
    # Obs[(0,1)] = [1]
    # Obs[(3,4)] = [1]
    # Obs[(4,5)] = [0]
    # Obs[(7,2)] = [0]
    # E_X = [(3,1), (6,1), (6,4), (7, 5)]

    # Obs[(7,4)] = [1]
    # Obs[(4,1)] = [1]
    # Obs[(4,5)] = [1]
    # Obs[(5,2)] = [1]
    # Obs[(1,2)] = [1]
    # Obs[(4,2)] = [0]
    # Obs[(7,2)] = [0]
    # E_X = [(7,2), (4,2), (7,5), (3, 1)]

    # Obs[(7,5)] = [0]

    # logging = Log()





"""
testcase1 will mainly focus on the evaluation of the prediction quality

testcase2 is designed to test if the algorithm runs without error in different settings.
"""
def testcase2():
    V = [0,1,2,3,4,5,6,7,8]
    E = [(0,1), (1,2), (3,4), (4,5), (6,7), (7,8), (3,0), (6,3), (4,1), (7,4), (5,2), (8,5)]
    datasets = simulation_data_generator(V, E, rates = [0.5, 0.1, 0.15, 0.2, 0.25, 0.50], realizations=10)
    logging = Log()
    for (rate, i), rate_dataset in datasets.items():
        [V, E, Omega, Obs, X] = rate_dataset
        evaluate(V, E, Obs, Omega, X, logging, psl = False, approx = False, init_alpha_beta = (1, 1))

def testcase3():
    graph_sizes = [500, 1000, 5000, 10000, 47676]
    rates = [0.05, 0.1, 0.15, 0.25, 0.3, 0.4, 0.5]
    realizations = 10
    logging = Log("testcase3.txt")
    for graph_size in graph_sizes[0:4]:
            for rate in rates[:1]:
                for real_i in range(realizations)[:1]:
                    filename = "data/trust-analysis/nodes-{0}-rate-{1}-realization-{2}-data.pkl".format(graph_size, rate, real_i)
                    if os.path.exists(filename):
                        starttime = time.time()
                        pkl_file = open(filename, 'rb')
                        [V, E, Omega, Obs, X] = pickle.load(pkl_file)
                        pkl_file.close()
                        logging.write('\r\n\r\nfilename: {0}, #nodes: {1}, #edges: {2}'.format(filename, len(V), len(E)))
                        # print "start evaluating"
                        evaluate(V, E, Obs, Omega, X, logging, psl = False, approx = True, init_alpha_beta = (1, 1))
                        endtime = time.time()
                        logging.write("{0}: {1} seconds".format(filename, endtime - starttime))

def testcase4():
    graph_sizes = [500, 1000, 5000, 10000, 47676]
    rates = [0.05, 0.1, 0.15, 0.25, 0.3, 0.4, 0.5]
    realizations = 10
    window_size = 8
    begin_time = 0
    end_time = 25
    logging = Log("testcase4.txt")
    for graph_size in graph_sizes[0:4]:
            for rate in rates:
                for real_i in range(realizations)[:3]:
                    filename = "data/trust-analysis/nodes-{0}-rate-{1}-realization-{2}-data.pkl".format(graph_size, rate, real_i)
                    if os.path.exists(filename):
                        running_starttime = time.time()
                        pkl_file = open(filename, 'rb')
                        [V, E, Omega, Obs, X] = pickle.load(pkl_file)
                        pkl_file.close()
                        logging.write('\r\n\r\nfilename: {0}, #nodes: {1}, #edges: {2}'.format(filename, len(V), len(E)))
                        sw_measures = []
                        for start_t in range(begin_time, end_time-window_size+1):
                            print "sliding window: {0} to {1}".format(start_t, start_t + window_size)
                            logging.write("\r\n sliding window: {0} to {1}".format(start_t, start_t + window_size))
                            sw_Omega, sw_Obs = sliding_window_extract(Omega, Obs, start_t, window_size)
                            pred_omega_x = inference_apdm_format(V, E, sw_Obs, sw_Omega, X, logging)
                            prob_mse, u_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested = calculate_measures(sw_Omega, pred_omega_x, X, logging)
                            sw_measures.append([prob_mse, u_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested])

                        avg_measures = [0, 0, 0, 0, 0, 0, 0]
                        for measures in sw_measures:
                            avg_measures = list_sum(avg_measures, measures)
                        avg_measures = list_dot_scalar(avg_measures, 1.0 / len(sw_measures))
                        logging.write("\r\n ----------------Summary of the results--------------------")
                        logging.write("prob_mse: {0}, u_mse: {1}, prob_relative_mse: {2}, u_relative_mse: {3}".format(prob_mse, u_mse, prob_relative_mse, u_relative_mse))
                        logging.write("accuracy: {0}, recall_congested: {1}, recall_uncongested: {2}".format(accuracy, recall_congested, recall_uncongested))

                        running_endtime = time.time()
                        logging.write("\r\n running time: {0} seconds".format(running_endtime - running_starttime))

                        # # print "start evaluating"
                        # evaluate(V, E, Obs, Omega, X, logging, psl = False, approx = True, init_alpha_beta = (1, 1))
                        # endtime = time.time()
                        # logging.write("{0}: {1} seconds".format(filename, endtime - starttime))



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
def evaluate(V, E, Obs, Omega, E_X, logging, psl = False, approx = False, init_alpha_beta = (1, 1), report_stat = False):
    # approx = False
    running_starttime = time.time()
    pred_omega_x = inference_apdm_format(V, E, Obs, Omega, E_X, logging, psl, approx, init_alpha_beta, report_stat)
    # prob_mse, u_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested
    alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested = calculate_measures(Omega, pred_omega_x, E_X, logging)
    # sw_measures, avg_measures = inference_apdm_format_sliding_window(V, E, Obs, Omega, E_X, window_size, logging, psl, approx, init_alpha_beta, report_stat)

    logging.write("\r\n ----------------Summary of the results--------------------")
    logging.write("prob_mse: {0}, u_mse: {1}, prob_relative_mse: {2}, u_relative_mse: {3}".format(prob_mse, u_mse, prob_relative_mse, u_relative_mse))
    logging.write("accuracy: {0}, recall_congested: {1}, recall_uncongested: {2}".format(accuracy, recall_congested, recall_uncongested))

    running_endtime = time.time()
    logging.write("\r\n running time: {0} seconds".format(running_endtime - running_starttime))

    return



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
V: a list of vertex ids
E: a list of ordered pairs of vertex ids
T: The number of observations. 100 by default.

OUTPUT
datasets: key is rate and the value is [[V, E, Omega, Obs, X], ...], where rate refers to the rate of edges that are randomly selected as target edges

"""
def simulation_data_generator(V, E, rates = [0.05, 0.1, 0.15, 0.25, 0.3, 0.4, 0.5], realizations = 10, T = 50):
    datasets = {}
    len_E = len(E)
    Omega = {}
    for e in E:
        Omega[e] = generate_a_random_opinion()
    for rate in rates:
        rate_datasets = []
        for real_i in range(realizations):
            len_X = int(round(len_E * rate))
            rate_X = [E[i] for i in np.random.permutation(len_X)[:len_X]]
            rate_omega_X = SL_prediction(V, E, Omega, rate_X)
            rate_omega = copy.deepcopy(Omega)
            for e, alpha_beta in rate_omega_X.items():
                Omega[e] = alpha_beta
            rate_Obs = {}
            for e in E:
                e_alpha, e_beta = Omega[e]
                rate_Obs[e] = beta.rvs(e_alpha, e_beta, 0, 1, T)
            datasets[(rate, real_i)] = [V, E, rate_omega, rate_Obs, rate_X]
    return datasets


"""
Generate simulation datasts with different graph sizes and rates
"""
def simulation_data_generator1():
    graph_sizes = [500, 1000, 5000, 10000, 47676]
    rates = [0.05, 0.1, 0.15, 0.25, 0.3, 0.4, 0.5]
    T = 10
    for graph_size in graph_sizes[4:5]:
        filename = "data/trust-analysis/nodes-{0}.pkl".format(graph_size)
        print "--------- reading {0}".format(filename)
        pkl_file = open("data/trust-analysis/nodes-{0}.pkl".format(graph_size), 'rb')
        [V, E] = pickle.load(pkl_file)
        print len(V), len(E)
        pkl_file.close()
        print "--------- generating simulation data"
        datasets = simulation_data_generator(V, E, rates[:1], T)
        for (rate, real_i), dataset in datasets.items():
            print "---------------------graph size: {0}, rate: {1}, realization: {2}".format(graph_size, rate, real_i)
            pkl_file = open("data/trust-analysis/nodes-{0}-rate-{1}-realization-{2}-data.pkl".format(graph_size, rate, real_i), 'wb')
            pickle.dump(dataset, pkl_file)
            pkl_file.close()



"""
Using breadfirst search with the start vertex id 0, this function generates sampled networks of
size 500, 1000, 5000, 10000, and 47676, where 47676 is the size of the largest connected component
that includes vertex 0
"""
def sampple_epinion_network(sample_sizes = [500, 1000, 5000, 10000, 47676]):
    filename = "data/trust-analysis/Epinions.txt"
    # print open(filename).readlines()[:4]
    dict_V = {}
    E = []
    for line in open(filename).readlines()[4:]:
        (str_start_v, str_end_v) = line.split()
        start_v = int(str_start_v)
        end_v = int(str_end_v)
        if not dict_V.has_key(start_v):
            dict_V[start_v] = 1
        if not dict_V.has_key(end_v):
            dict_V[end_v] = 1
        E.append((start_v, end_v))
    V = dict_V.keys()
    vertex_nns = {}
    for v_start, v_end in E:
        if vertex_nns.has_key(v_start):
            if v_end not in vertex_nns[v_start]:
                vertex_nns[v_start].append(v_end)
        else:
            vertex_nns[v_start] = [v_end]

    sample_networks = []
    for sample_size in sample_sizes:
        sample_V, sample_E = breadth_first_search(vertex_nns, sample_size)
        print "sample network: #nodes: {0}, #edges: {1}".format(len(sample_V), len(sample_E))
        # print walk_V
        # print walk_E
        sample_networks.append([sample_V, sample_E])
        pkl_file = open("data/trust-analysis/nodes-{0}.pkl".format(len(sample_V)), 'wb')
        pickle.dump([sample_V, sample_E], pkl_file)
        pkl_file.close()



"""
INPUT
vertex_nns: key is vertex id, and the value is a list of neighboring vertex ids
sample_size: the size of the sampled subgraph using breadth first search

We need to re-label the vertex ids of the sampled subgraph such that the ids are continuous starting from 0.

OUTPUT
V: a list of vertex ids
E: a list of pairs of vertex ids
"""
def breadth_first_search(vertex_nns, sample_size):
    sample_V = {}
    sample_E = []
    start_v = 0
    queue = [start_v]
    n = 0
    while len(queue) > 0:
        v = queue.pop()
        sample_V[v] = 1
        n = n + 1
        if vertex_nns.has_key(v):
            for v_n in vertex_nns[v]:
                if not sample_V.has_key(v_n) and v_n not in queue:
                    queue.append(v_n)
        if n >= sample_size:
            break
    for v, v_nns in vertex_nns.items():
        for v_n in v_nns:
            if sample_V.has_key(v) and sample_V.has_key(v_n):
                sample_E.append((v, v_n))
    old_id_2_new_id = {}
    for new_id, v in enumerate(sample_V.keys()):
        old_id_2_new_id[v] = new_id
    sample_V = range(len(sample_V))
    sample_E = [(old_id_2_new_id[v1], old_id_2_new_id[v2]) for (v1, v2) in sample_E]
    return sample_V, sample_E


# def random_walk(nns, walk_len, restart_prob = 0.02):
#     walk_V = {}
#     walk_E = {}
#     len_E = len(nns)
#     start_v = round(random.random() * len_E)
#     cur_v = start_v
#     while(len(walk_V) < walk_len):
#         walk_V[cur_v] = 1
#         if not nns.has_key(cur_v):
#             cur_v = start_v
#             continue
#         next_v = random.choice(nns[cur_v])
#         walk_E[(cur_v, next_v)] = 1
#         cur_v = next_v
#         if random.random() < restart_prob:
#             cur_v = start_v
#     return walk_V.keys(), walk_E.keys()

def main():
    testcase6()
    # simulation_data_generator1()
    # sampple_epinion_network()
    return
    # print np.arange(0, 1, 0.01)
    testcase2()
    # E_X = [(2, 7), (7, 12), (15, 16), (18, 19)]
    # E_X = [(2, 7), (7, 12)]
    # for i in range(1, 101):
    #     filename = "APDM-GirdData-k{0}.txt".format(i)
    #     print "-----------------------{0}".format(filename)
    #     V, E, Obs, Omega = readAPDM(filename)
    #     evaluate(V, E, Obs, Omega, E_X)
    #     # break

if __name__=='__main__':
    main()
