__author__ = 'Feng Chen'
from math import *
import numpy as np
from scipy.stats import gamma
# import cvxpy as cvx
# from cvxpy import *
# import cvxopt
import copy
import random, math

import os
import pickle
import sys
import time
# from readAPDM import readAPDM
import scipy
# scipy.stats.
# import scipy.stats
from scipy.stats import beta
from cubic import cubic
from log import Log
import time

time.time()
import re
import warnings
import multiprocessing,inspect


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


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
    id_2_edge = {0: (1, 2), 1: (1, 2), 2: (0, 2)}
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
    for edge_id, (v1, v2) in edge_id_map.items():
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


def inference_admm(omega, b, y_t, Y, X, node_nns, omega_0, R,v0, psl, approx, report_stat):

    p_adv= admm(omega, b, y_t, Y, X, node_nns,
                    omega_0, R,v0, psl, approx, report_stat)
    return p_adv




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


def inference(omega_y, omega_0, y, X, Y, T, node_nns, R,v0, logging, psl=False, approx=False,
              init_alpha_beta=(1, 1), report_stat=False):
    omega = copy.deepcopy(omega_y)
    for v in X.keys(): omega[
        v] = init_alpha_beta  # Beta pdf can be visualized via http://eurekastatistics.com/beta-distribution-pdf-grapher/
    error = -1
    for iter in range(1):
        ps = []
        bs = []

        b = {v: 0 for v in X.keys() + Y.keys()}
        y_t = {v: v_y[0] for v, v_y in y.items()}
        p_adv=inference_admm(omega, b, y_t, Y, X, node_nns, omega_0, R,v0, psl, approx, report_stat)
            # print "b_t", b_t


    return p_adv


"""
INPUT
bs: a list of dictionaries. Each dictionary (b) in the list is defined as follows:
    key is an edge id and value is the binary value of this edge indicating if this edge is compromised or not.

OUTPUT
b_new: a single dictionary (b) that is estimated based on bs.
"""



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


def inference_apdm_format(V, E, Obs, Omega, b, X,v0, logging, psl=False, approx=True, init_alpha_beta=(1, 1),
                          report_stat=False):
    node_nns, id_2_node, node_2_id, omega, feat = reformat_nodes(V, E, Obs, Omega)
    test_list = []
    X = {node_2_id[v]: 1 for v in X}
    v0 = node_2_id[v0]
    Y = {v: 1 for v in id_2_node.keys() if not X.has_key(v)}
    # b = {edge_2_id[e]: val for e, val in b.items()}
    omega_y = {v: omega[v] for v in Y}
    y = {v: feat[v] for v in Y}
    # omega_0 = [2, 15]
    omega_0 = [2, 8]
    # omega_0 = [15, 2]
    T = len(feat[0])
    R = generate_PSL_rules_from_edge_cnns_nodes(node_nns)
    print "#Rule:{} Nodes:{} Edges:{}".format(len(R), len(V), len(E))
    p_adv= inference(omega_y, omega_0, y, X, Y, T, node_nns, R,v0, logging, psl,
                                       approx, init_alpha_beta, report_stat)
    # sign_grad_py = []
    # for sign_grad_py_t in sign_grad_py_ids:
    #     temp_dic = {}
    #     for id, sign in sign_grad_py_t.items():
    #         temp_dic[id_2_node[id]] = sign
    #     sign_grad_py.append(temp_dic)

    return p_adv



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
        sw_Obs[e] = [Obs[e][t] for t in range(start_t, start_t + window_size)]
        n = np.sum(sw_Obs[e])
        sw_Omega[e] = (n + 0.001, window_size - n + 0.001)
    return sw_Omega, sw_Obs


"""
INPUT
p: [p_1, p_2, ... p_T]. p_t is a list of truth probability values of all edges. p[t][i] refers to the probability value of edge id=i at time t, where t \in {1, ..., T} and i \in {1, cnt_Y+cnt_X}
omega: A dictionary of opinions (tuples of alpha and beta values) of all edges (X+Y): [edge1 id: [alpha_1, beta_1], ...]. Note that the opinions related to the edges in X will be udpated based on p.
X: a list of edge ids whose opinions need to be predicted

OUTPUT
omega_x: A dictionary of opinions (tuples of alpha and beta values): [edge1 id: [alpha_1, beta_1], ..., edge_N id: [alpha_M, beta_N]]. cnt_X is the size of X.
"""
# def estimate_omega_x(p, omega, X):
#     for e in X:
#         data = [p_t[e] for p_t in p]
#         if np.std(data) < 0.01:
#             alpha1 = np.mean(data)
#             beta1 = 1 - alpha1
#         else:
#             alpha1, beta1, loc, scale = beta.fit(data, floc=0,fscale=1)
#         alpha1 = alpha1 / (alpha1 + beta1) * 10
#         beta1 = 10 - alpha1
#         omega[e] = (alpha1, beta1)
#     return omega
"""
INPUT
p: [p_1, p_2, ... p_T]. p_t is a list of truth probability values of all edges. p[t][i] refers to the probability value of edge id=i at time t, where t \in {1, ..., T} and i \in {1, cnt_Y+cnt_X}
omega: A dictionary of opinions (tuples of alpha and beta values) of all edges (X+Y): [edge1 id: [alpha_1, beta_1], ...]. Note that the opinions related to the edges in X will be udpated based on p.
X: a list of edge ids whose opinions need to be predicted

OUTPUT
omega_x: A dictionary of opinions (tuples of alpha and beta values): [edge1 id: [alpha_1, beta_1], ..., edge_N id: [alpha_M, beta_N]]. cnt_X is the size of X.
"""


def estimate_omega_x(p, omega, X):
    for e in X:
        data = [p_t[e] for p_t in p]
        if np.std(data) < 0.01:
            alpha1 = np.mean(data)
            beta1 = 1 - alpha1
        else:
            data = [max([p_t[e] - random.random() * 0.01, 0]) for p_t in p]
            try:
                alpha1, beta1, loc, scale = beta.fit(data, floc=0., fscale=1.)
                if alpha1 < 1:
                    beta1 = 1.1 * beta1 / alpha1
                    alpha1 = 1.1
                if beta1 + alpha1 > 10:
                    alpha1 = alpha1 / (alpha1 + beta1) * 10
                    beta1 = 10 - alpha1
            except:
                alpha1 = 1
                beta1 = 1
            print alpha1, beta1
            # print alpha1, beta1
        omega[e] = (alpha1, beta1)
    return omega


def estimate_omega_x(ps, X):
    omega_x = {}
    strategy = 1  # 1: means we consider p values as binary observations and use them to estimate alpha and beta.
    if strategy == 1:
        for e in X:
            data = [prob_2_binary2(p_t[e]) for p_t in ps]
            # data = [(p_t[e]) for p_t in ps]
            alpha1 = np.sum(data) + 1.0
            beta1 = len(data) - np.sum(data)  + 1.0
            omega_x[e] = (alpha1, beta1)
    return omega_x


def estimate_b(bs):
    cnt_Y = len(bs[0])
    b_new = {}
    for i in range(cnt_Y):
        b_new[i] = prob_2_binary(np.mean([b_t[i] for b_t in bs]))
    return b_new


def prob_2_binary(val):
    return val
    # if val > 0.45:
    #     return 1
    # else:
    #     return 0

def prob_2_binary2(val):
    # return val
    if val >= 0.2:
        return 1
    else:
        return 0

def list_dot_scalar(l, c):
    return [item * c for item in l]


def list_sum(l1, l2):
    l = []
    for v1, v2 in zip(l1, l2):
        l.append(v1 + v2)
    return l

def list_inn(l1, l2):
    l = 0.0
    for v1, v2 in zip(l1, l2):
        l += v1 * v2
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


"""
INPUT
edge_nns: key is an edge id, and its value is a list of its neighbor edge ids.

OUTPUT
R: a list of pairs of neighboring edge ids.
"""


def generate_PSL_rules_from_edge_cnns(edge_up_nns, edge_down_nns):
    dic_R = {}
    e_checked = {}
    for e, neighbors in edge_up_nns.items():
        for up_e in neighbors.keys():
            if not dic_R.has_key((e,
                                  up_e)):  # The direction should be from e_prev to e. The traffic at e will impact the traffic of its up adjacent edges
                dic_R[(e, up_e)] = 1
                # e_checked[e] = 1
                # e_checked[up_e] = 1
    # for e, neighbors in edge_down_nns.items():
    #     if not e_checked.has_key(e) and len(neighbors.keys())>0:
    #         print "...................."
    #         dic_R[(e, neighbors.keys()[0])]=1
    return dic_R.keys()

def generate_PSL_rules_from_edge_cnns_nodes(node_nns):
    dic_R = {}
    for v, neighbors in node_nns.items():
        for n_v in neighbors:
            if not dic_R.has_key((v, n_v)):  # The direction should be from up_e to e. The traffic at e will impact the traffic of its up adjacent edges
                dic_R[(v, n_v)] = 1
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
rho: step size. rho=kappa

OUTPUT
R_p: The probability values in R_p are updated based on R_z and R_lambda_.

"""


def update_px(X, R_p, R_p_hat, R_lambda_, copies, omega, cnt_V, kappa, approx):
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
        aa = -1 * (1 / kappa) * nc
        z_lambda_sum = sum([R_p_hat[k][j] + kappa * R_lambda_[k][j] for k, j in copies[e]])
        bb = (1 / kappa) * (nc + z_lambda_sum)
        cc = omega[e][0] + omega[e][1] - 2 - (1 / kappa) * z_lambda_sum
        dd = 1 - omega[e][0]

        if approx:
            min_prob = sol_min_p_approx(e, omega, R_p_hat, R_lambda_, kappa, copies, aa, bb, cc, dd)
        else:
            min_prob = sol_min_p(e, omega, R_p_hat, R_lambda_, kappa, copies, aa, bb, cc, dd)

        for k, j in copies[e]:
            R_p[k][j] = min_prob
    return R_p
    #                sign_grad_py_t,p_t,b_t,y_t,Y, R_p, R_p_hat, R_lambda, copies, omega,kappa, appro
def get_sign_grad_py(sign_grad_py_t,p_t,b_t,y_t,Y, R_p, R_p_hat, R_lambda_, copies, omega, kappa, approx):
    # py_grad_sign={}
    grads={}
    for v in Y.keys():
        grad=0
        """
        grad_i= - y_t[e]/p_y[e] +(1-y_t[e])/(1-p_y[e]) + rho*\sum_{copies p[e]}(p_t[e]-\hat p[e] -\lambda[e]*1/rho )
        """
        if not copies.has_key(v):
            if sign_grad_py_t.has_key(v):
                sign_grad_py_t[v].append(0.0)  #the edges  are not involved any rules
            else:
                sign_grad_py_t[v]=[0.0]
            continue

        nc = len(copies[v])
        #\alpha_0==\alpha_e
        # C1= omega[v][0] - 1 + y_t[v]
        # C2= omega[v][1] - y_t[v]

        # C1 = (1-b_t[v])*(omega[v][0] - 1) + y_t[v]
        # C2 =(1-b_t[v])*(omega[v][1] - 1) +1 - y_t[v]
        if p_t[v]==0.0:
            term1=0.0
        else:
            term1=-1.0

        if p_t[v]==1.0:
            term2=0.0
        else:
            term2=1.0
        # C1= y_t[v]
        # C2 = 1 - y_t[v]
        z_lambda_sum = np.sum([R_p[v]-R_p_hat[k][j] - kappa * R_lambda_[k][j] for k, j in copies[v]])
        # print v,[(p_t[v],R_p_hat[k][j],R_lambda_[k][j],p_t[v]-R_p_hat[k][j] - kappa * R_lambda_[k][j]) for k, j in copies[v]]
        # if z_lambda_sum!=0:
        #     print ">>>>>>>>",p_t[e],z_lambda_sum
        #    -C1 / (p_t[v] + 0.000001) + C2 / (1 - p_t[v]

        grad=np.sign(term1 +term2  +(1/kappa)* z_lambda_sum)
        if b_t[v]==1.0 : grad=0.0
        # grad=z_lambda_sum
        if grads.has_key(grad):
            grads[grad]+=1
        else:
            grads[grad] = 1
        # print grad
        if grad == 0:
            if sign_grad_py_t.has_key(v):
                sign_grad_py_t[v].append(0.0)
            else:
                sign_grad_py_t[v] = [0.0]

        elif grad > 0:
            if sign_grad_py_t.has_key(v):
                sign_grad_py_t[v].append(1.0)
            else:
                sign_grad_py_t[v] = [1.0]
        else:
            if sign_grad_py_t.has_key(v):
                sign_grad_py_t[v].append(-1.0)
            else:
                sign_grad_py_t[v] = [-1.0]
        # time.sleep(100)
    print(grads)
    return sign_grad_py_t

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
rho: step size. rho=kappa

OUTPUT
R_p: The probability values in R_p are updated based on R_z and R_lambda_.

            prob = normalize_prob(b_y_lambda_sum / nc)
            for k, j in copies[e]:
                R_p[k][j+1] = prob
"""


def update_by(p, y_t, Y, R_p, R_p_hat, R_lambda_, copies, omega, omega_0, cnt_V, kappa, approx):
    for e in Y:
        # p0 = y_t[e] + 0.001 if y_t[e] == 0 else y_t[e] - 0.001
        if not copies.has_key(e): continue
        nc = len(copies[e])
        term1 = (1.0 * kappa / nc)
        by_lambda_sum = sum([R_p_hat[k][j + 1] + kappa * R_lambda_[k][j + 1] for k, j in copies[e]])
        # print "omega", omega
        # print omega[e]
        # print np.power(p[e],omega[e][0]-1), omega[e][0]-1
        # omega[e] = [2, 10]
        # omega_0 = [2, 10]
        # print "omega[e]: omega_0, p[e]", omega[e], omega_0, p[e]
        omega_0 = omega[e]
        p0 = 0.5
        # C_3 = np.log(max(0.0001, gamma(omega[e][0]+omega[e][1]))) - np.log(max(0.0001, gamma(omega[e][0]))) - np.log(max(0.0001, gamma(omega[e][1]))) + \
        #       (omega[e][0]-1) * np.log(max(0.0001, p[e])) + (omega[e][1]-1) * np.log(max(0.0001, 1 - p[e]))
        # # print "1", np.log(np.power(p[e],omega_0[0]-1)) - np.log(max(0.0001, np.power(1-p[e],omega_0[1]-1))))
        # C_4 = np.log(max(0.0001, gamma(omega_0[0]+omega_0[1])))   - np.log(max(0.0001, gamma(omega_0[0])))  - np.log(max(0.0001, gamma(omega_0[1]))) + \
        #       (omega_0[0]-1)  * np.log(max(0.0001, p[e])) + (omega_0[1]-1) * np.log(max(0.0001, 1 - p[e]))
        # # print C_3, C_4
        # term2 = np.log(max(p0, 0.0001)) - np.log(max(1 - p0, 0.0001))
        C_3 = 0.0
        C_4 = 0.0
        term2 = 0.0
        prob = normalize_prob(term1 * (- C_3 + C_4 + term2 + (1.0 / kappa) * by_lambda_sum))
        # print "prob, C_3, C_4", prob, C_3, C_4, C_4 - C_3, term2, (1.0/kappa)*by_lambda_sum
        # prob = term1*( (1.0/kappa)*by_lambda_sum )
        # if b_y > 1: b_y = 1
        # elif b_y < 0:
        #     b_y = 0
        for k, j in copies[e]:
            R_p[k][j + 1] = prob
    return R_p


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
        score = -1 * (omega_e[0] - 1) * log(prob) - (omega_e[1] - 1) * log(1 - prob) + rho * 0.5 * sum(
            [pow(prob - dict_R_z[k][j] - dict_R_lambda_[k][j] / rho, 2) for k, j in copies_e])
        if score < min_score:
            min_prob = prob
            min_score = score
    if min_prob == -1:
        debug = 1
    return min_prob


def sol_min_p(e, omega, R_z, R_lambda_, kappa, copies, aa, bb, cc, dd):
    probs = cubic(aa, bb, cc, dd)
    probs = [prob for prob in probs if type(prob) is not complex and prob > 0 and prob < 1]
    probs.extend([0.001, 0.999])
    min_prob = -1
    min_score = float('inf')
    for prob in probs:
        score = -1 * (omega[e][0] - 1) * np.log(prob) - (omega[e][1] - 1) * np.log(1 - prob) + (1 / kappa) * 0.5 * sum(
            [pow(prob - R_z[k][j] - kappa * R_lambda_[k][j], 2) for k, j in copies[e]])
        # print "out",type(score),score ,min_score
        if score < min_score:
            # print "in",type(score) ,min_score
            min_prob = prob
            min_score = score
    if min_prob == -1:
        debug = 1

    return min_prob


def sol_min_p_approx1(e, omega_e, dict_R_z, dict_R_lambda_, rho, copies_e, aa, bb, cc, dd):
    min_prob = -1
    min_score = float('inf')
    for prob in np.arange(0.01, 1, 0.01):
        score = -1 * (omega_e[0] - 1) * log(prob) - (omega_e[1] - 1) * log(1 - prob) + rho * 0.5 * sum(
            [pow(prob - dict_R_z[k][j] - dict_R_lambda_[k][j] / rho, 2) for [k, j] in copies_e])
        if score < min_score:
            min_prob = prob
            min_score = score
    if min_prob == -1:
        debug = 1
    return min_prob


def sol_min_p_approx(e, omega, R_z, R_lambda_, rho, copies, aa, bb, cc, dd):
    min_prob = -1
    min_score = float('inf')
    # for prob in np.arange(0.01, 1, 0.01):
    probs = list(np.arange(0.1, 1, 0.1))
    probs.extend([0.001, 0.01, 0.99, 0.999])
    for prob in probs:
        score = -1 * (omega[e][0] - 1) * np.log(prob) - (omega[e][1] - 1) * np.log(1 - prob) + rho * 0.5 * sum(
            [pow(prob - R_z[k][j] - R_lambda_[k][j] / rho, 2) for k, j in copies[e]])

        if score < min_score:
            min_prob = prob
            min_score = score
    if min_prob == -1:
        debug = 1
    return min_prob


def normalize_prob(prob):
    if prob < 0: prob = 0
    if prob > 1: prob = 1
    return prob


"""
It will only update p_x and b_y, but not p_y
"""


def psl_update_px_by(Y, R_p, R_p_hat, R_lambda_, copies, omega, cnt_V, kappa):
    for v in range(cnt_V):
        if v not in Y:  # update p_x
            if not copies.has_key(v): continue
            nc = len(copies[v]) * 1.0
            p_x_lambda_sum = sum([R_p_hat[k][j] + R_lambda_[k][j] * kappa for k, j in copies[v]])
            prob = normalize_prob(p_x_lambda_sum / nc)
            for k, j in copies[v]:
                R_p[k][j] = prob
        else:  # update b_y
            if not copies.has_key(v): continue
            nc = len(copies[v]) * 1.0
            b_y_lambda_sum = sum([R_p_hat[k][j + 1] + R_lambda_[k][j + 1] * kappa for k, j in copies[v]])
            prob = normalize_prob(b_y_lambda_sum / nc)
            for k, j in copies[v]:
                R_p[k][j + 1] = prob
    return R_p


"""
INPUT
y_t: A dictionary of lists of observations at time t with key an edge id and value its observation at time t.
Y: The subset of edge ids that have known opinions
edge_nns: key is an edge id, and its value is a list of its neighbor edge ids.
cnt_V: The total number of edges
cnt_X: the number of edges whose opinions will be predicted (size of X).
p0: The prior probability that an edge is congested. It is calculated the ratio of total number of congested links indexed by link id and time over the total number of links.
OUTPUT
p: a list of probability initial values of the cnt_Y + cnt_X edges. p[i] refers to the initial probability value of edge id=i
"""


# def calc_initial_p(y_t, edge_down_nns, X, Y, cnt_V, p0):
#     p = [1e-6 for i in range(cnt_V)]
#     for e, e_nns in edge_down_nns.items():
#         if X.has_key(e):
#             obs = [y_t[e_n] for e_n in e_nns.keys() if Y.has_key(e_n)]
#             if len(obs) == 0: obs = [p0]
#             p[e] = np.median(obs)
#         else:
#             p[e] = y_t[e]
#     for i in Y:
#         p[i] = y_t[i]
#     return p

def calc_initial_p(y_t, edge_down_nns, X, Y, cnt_V, p0, b_init):
    p = [1e-6 for i in range(cnt_V)]

    for e in Y:
        p[e] = y_t[e]  # observation indicates probability
        """ dict_paths[e] includes the rule bodies that indicates e """
        # if dict_paths.has_key(e) and X_b.has_key(e):
        n_pos = 0
        n_neg = 0
        for e, e_nns in edge_down_nns.items():
            obs = [y_t[e_n] for e_n in e_nns.keys() if Y.has_key(e_n)]
            for val in obs:
                if val > 0:
                    n_pos += 1.0
                else:
                    n_neg += 1.0
            if (n_pos - n_neg > 0 and p[e] == 0) or (n_pos - n_neg < 0 and p[e] > 0):
                b_init[e] = 1.0
            else:
                b_init[e] = 0.0

    for e, e_nns in edge_down_nns.items():
        if X.has_key(e):
            obs = [y_t[e_n] for e_n in e_nns.keys() if Y.has_key(e_n)]
            if len(obs) == 0: obs = [p0]
            n_pos = 0
            n_neg = 0
            for val in obs:
                if val > 0:
                    n_pos += 1.0
                else:
                    n_neg += 1.0
            if n_pos > n_neg:
                p[e] = 1.0
            else:
                p[e] = 0.0
            # p[e] = np.median(obs)
        else:
            p[e] = y_t[e]
    # for i in Y:
    #     if b_init[i]==1.0:
    #         p[i] = 1.0-y_t[i]
    #     else:
    #         p[i] = y_t[i]

    return p, b_init

def calc_initial_p1(y_t, edge_down_nns, X, Y, cnt_V, p0, b_init):
    p = [1e-6 for i in range(cnt_V)]

    for e in Y:
        p[e] = y_t[e]  # observation indicates probability
        """ dict_paths[e] includes the rule bodies that indicates e """
        # if dict_paths.has_key(e) and X_b.has_key(e):
        n_pos = 0
        n_neg = 0
        for e, e_nns in edge_down_nns.items():
            obs = [y_t[e_n] for e_n in e_nns.keys() if Y.has_key(e_n)]
            for val in obs:
                if val > 0:
                    n_pos += 1.0
                else:
                    n_neg += 1.0
            if (n_pos - n_neg > 0 and p[e] == 0) or (n_pos - n_neg < 0 and p[e] > 0):
                b_init[e] = 1.0
            else:
                b_init[e] = 0.0

    for e, e_nns in edge_down_nns.items():
        if X.has_key(e):
            obs = {e_n:y_t[e_n] for e_n in e_nns.keys() if Y.has_key(e_n)}
            # if len(obs) == 0: obs = {p0}
            conf = 0
            n_pos=0.0
            for e_o,val in obs.items():
                # if Y.has_key(e_o) and b_init[e_o]==1.0:
                #     val=np.abs(1.0-val)
                n_pos+=val
                if val > 0:
                    conf += 1.0
                else:
                    conf -= 1.0
            if conf > 0:
                p[e] = 1.0
            else:
                # p[e] = 0.0
                if n_pos>1:
                    p[e] = 1.0
                else:
                    p[e] = 0.0
        else:
            p[e] = y_t[e]

    return p, b_init

def calc_initial_node_p1(y_t, node_nns, X, Y, cnt_V, p0, b_init):
    p = [1e-6 for i in range(cnt_V)]
    # print "Y:{} X:{} node_nns:{}".format(len(Y),len(X),len(node_nns))
    for v in Y:
        p[v] = y_t[v]  # observation indicates probability
        """ dict_paths[e] includes the rule bodies that indicates e """
        # if dict_paths.has_key(e) and X_b.has_key(e):
        n_pos = 0
        n_neg = 0

        obs = [y_t[v_n] for v_n in node_nns[v] if Y.has_key(v_n)]
        for val in obs:
            if val > 0:
                n_pos += 1.0
            else:
                n_neg += 1.0
        if (n_pos - n_neg > 0 and p[v] == 0) or (n_pos - n_neg < 0 and p[v] > 0):
            b_init[v] = 1.0
        else:
            b_init[v] = 0.0

    for v, v_nns in node_nns.items():
        if X.has_key(v):
            obs = {v_n:y_t[v_n] for v_n in v_nns if Y.has_key(v_n)}
            # if len(obs) == 0: obs = {p0}
            conf = 0
            n_pos=0.0
            for v_o,val in obs.items():
                # if Y.has_key(e_o) and b_init[e_o]==1.0:
                #     val=np.abs(1.0-val)
                n_pos+=val
                if val > 0:
                    conf += 1.0
                else:
                    conf -= 1.0
            if conf > 0:
                p[v] = 1.0
            else:
                p[v] = 0.0
                # if n_pos>0.0:
                #     p[v] = 1.0
                # else:
                #     p[v] = 0.0
        else:
            p[v] = y_t[v]

    return p, b_init

def calc_initial_node_p2(y_t, node_nns, X, Y, cnt_V, p0, b_init):
    p = [1e-6 for i in range(cnt_V)]
    # print "Y:{} X:{} node_nns:{}".format(len(Y),len(X),len(node_nns))
    for v in Y:
        p[v] = y_t[v]  # observation indicates probability
        """ dict_paths[e] includes the rule bodies that indicates e """
        # if dict_paths.has_key(e) and X_b.has_key(e):
        n_pos = 0
        n_neg = 0

        obs = [y_t[v_n] for v_n in node_nns[v] if Y.has_key(v_n)]
        for val in obs:
            if val > 0.5:
                n_pos += 1.0
            else:
                n_neg += 1.0
        if (n_pos - n_neg > 0 and p[v] < 0.5) or (n_pos - n_neg < 0 and p[v] >= 0.5):
            b_init[v] = 1.0
        else:
            b_init[v] = 0.0

    for v, v_nns in node_nns.items():
        if X.has_key(v):
            obs = {v_n:y_t[v_n] for v_n in v_nns if Y.has_key(v_n)}
            # if len(obs) == 0: obs = {p0}
            conf = 0
            n_pos=0.0
            for v_o,val in obs.items():
                # if Y.has_key(e_o) and b_init[e_o]==1.0:
                #     val=np.abs(1.0-val)
                n_pos+=val
                # if b_init[v_o]==1:
                #     val=1.0-val
                if val > 0.5:
                    conf += 1.0
                else:
                    conf -= 1.0
            if conf > 0:
                p[v] = 1.0
            else:
                p[v] = 0.0
                # if n_pos>0.0:
                #     p[v] = 1.0
                # else:
                #     p[v] = 0.0
        else:
            p[v] = y_t[v]

    return p, b_init

def calc_initial_node_p3(y_t, node_nns, X, Y, cnt_V, p0, b_init):
    p = [1e-6 for i in range(cnt_V)]
    # print "Y:{} X:{} node_nns:{}".format(len(Y),len(X),len(node_nns))
    for v in Y:
        p[v] = y_t[v]  # observation indicates probability
        """ dict_paths[e] includes the rule bodies that indicates e """
        # if dict_paths.has_key(e) and X_b.has_key(e):
        n_pos = 0
        n_neg = 0

        obs = [y_t[v_n] for v_n in node_nns[v] if Y.has_key(v_n)]
        for val in obs:
            if val > 0:
                n_pos += 1.0
            else:
                n_neg += 1.0
        if (n_pos - n_neg > 0 and p[v] == 0) or (n_pos - n_neg < 0 and p[v] > 0):
            b_init[v] = 1.0
        else:
            b_init[v] = 0.0

    for v, v_nns in node_nns.items():
        if X.has_key(v):
            obs = {v_n:y_t[v_n] for v_n in v_nns if Y.has_key(v_n)}
            # if len(obs) == 0: obs = {p0}
            conf = 0
            n_pos=0.0
            for v_o,val in obs.items():
                # if Y.has_key(e_o) and b_init[e_o]==1.0:
                #     val=np.abs(1.0-val)
                n_pos+=val
                if b_init[v_o]==1:
                    val=1.0-val
                if val > 0:
                    conf += 1.0
                else:
                    conf -= 1.0
            if conf > 0:
                p[v] = 1.0
            else:
                p[v] = 0.0
                # if n_pos>0.0:
                #     p[v] = 1.0
                # else:
                #     p[v] = 0.0
        else:
            val=y_t[v]
            if b_init[v] == 1:
                val = 1.0 - val
            p[v] = val

    return p, b_init

def calc_initial_p2(y_t, edge_nns, Y, cnt_V, p0):
    p = [0 for i in range(cnt_V)]
    for e, e_nns in edge_nns.items():
        obs = [y_t[e_n] for e_n in e_nns if e_n in Y]
        if len(obs) == 0: obs = [p0]
        p[e] = np.mean(obs)
    return p


"""
INPUT
v: a vector of numbers that may be outside the range [0, 1]

OUTPUT
v: truncated vector of numbers in the range [0, 1]
"""


def normalize(v):
    for i in range(len(v)):
        if v[i] < 0.0: v[i] = 0.0
        if v[i] > 1.0: v[i] = 1.0
    return v


"""
INPUT

OUTPUT
"""


def generate_local_variables(p_init, b_init, X, R):
    K = len(R)
    R_p = np.zeros((K, 4))
    R_p_hat = np.zeros((K,4))  # R_z is a vector of lists of copied variables in R_p and each of the list relates to a specific rule.  R_z[i][e] is a copied variable of R_p[i][e]
    R_lambda = np.zeros((K, 4))
    y_edge = np.zeros((K, 2))
    copies = {}
    for k in xrange(K):
        for idx, v in enumerate(R[k]):  # R=[..,(e,e_up)_k,...]   e_up => e
            if X.has_key(v):
                R_p_hat[k][2 * idx] = p_init[v]  # copy p_x entry
                # copy b_y entry
                R_p[k][2 * idx] = p_init[v]  # p_x entry
            else:
                R_p_hat[k][2 * idx] = p_init[v]
                R_p_hat[k][2 * idx + 1] = b_init[v]
                R_p[k][2 * idx] = p_init[v]
                R_p[k][2 * idx + 1] = b_init[v]
                y_edge[k][idx] = 1.0  # indicates the edge in E_Y


            if copies.has_key(v):
                copies[v].append([k, idx * 2])
            else:
                copies[v] = [[k, idx * 2]]
        # print R_p_hat[k]


    return R_p, R_p_hat, R_lambda, y_edge, copies


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
p_t: a list of probability values of the cnt_Y + cnt_X edges. p[i] refers to the probability value of edge id=i
b_t:
"""

def admm(omega, b, y_t, Y, X, node_nns, omega_0, R,v0, psl=False, approx=False, report_stat=False):
    # print "b_init", b_init
    # print("v0",v0)
    weight = 1.0
    epsilon = 0.001
    cnt_V = len(X) + len(Y)
    K = len(R)
    t0 = time.time()
    sign_grad_py_t={}

    p_init, b_init = calc_initial_node_p2(y_t, node_nns, X, Y, cnt_V, 0.5, b)
    t1 = time.time()
    # print "cal int time",t1-t0
    # print "R", R
    # print "y_t", y_t
    # print "p_init", p_init
    R_p, R_p_hat, R_lambda, y_edge, copies = generate_local_variables(p_init, b_init, X,R)
    t2=time.time()
    # print "init var time",t2-t1
    # print R_p, copies
    kappa = 1.0  # kappa = 1/rho
    maxiter = 5
    for iter in range(maxiter):
        # print "admm iteration {}".format(iter)
        t3 = time.time()
        t_1 = time.time()
        proj_num=0
        for k in range(K):
            # update lambda variables
            #R_lambda[k] = list_sum(R_lambda[k], list_dot_scalar(list_minus(R_p_hat[k], R_p[k]), 1 / kappa))
            R_lambda[k] = R_lambda[k] + (1.0 / kappa) * (R_p_hat[k] - R_p[k])
            # update copy variables, assume lk(R_p_hat[k])<0
            # R_p_hat[k] = normalize(list_minus(R_p[k], list_dot_scalar(R_lambda[k], kappa)))
            R_p_hat[k] = normalize(R_p[k] - kappa * R_lambda[k])
            if lk(R_p_hat[k], y_edge[k]) > 0:
                c = delta_lk(R_p_hat[k], y_edge[k])
                c = np.array(c)
                # print "R_p_hat[k] before: ", R_p_hat[k]
                # k_p_hat = normalize(list_minus(R_p[k], list_sum(list_dot_scalar(R_lambda[k], kappa),
                #                                                 list_dot_scalar(c, weight * kappa))))
                k_p_hat = normalize(R_p[k] -(kappa*R_lambda[k]+weight * kappa*c))
                # print R_p_hat[k]
                # print "k_p_hat", k_p_hat
                for i, edge_i_val in enumerate(y_edge[k]):
                    if y_edge[k][i] == 1:  # update b_y
                        R_p_hat[k][i * 2 + 1] = k_p_hat[i * 2 + 1]
                    else:  # update p_x
                        R_p_hat[k][i * 2] = k_p_hat[i * 2]
                if lk(R_p_hat[k], y_edge[k]) < 0:
                    proj_num+=1
                    # print "proj k......"
                    # print "projection: ", lk(R_p_hat[k],y_edge[k]), R_p_hat[k]  #(Y,R_k,x, rule, y_edge)
                    R_p_hat[k] = normalize(Proj_lk2(R_p[k] - kappa * R_lambda[k], R_p_hat[k], y_edge[k]))
                # print "R_p_hat[k] updated: ", R_p_hat[k]
        t_2 = time.time()
        # update probability variables
        p_t_old, b_t_old = R_p_2_p(R_p, copies, cnt_V)
        # print "y_edge", y_edge
        # R_p_temp = psl_update_px_by(Y, R_p, R_p_hat, R_lambda, copies, omega, cnt_V, kappa)
        if psl == True:
            R_p = psl_update_px_by(Y, R_p, R_p_hat, R_lambda, copies, omega, cnt_V, kappa)
        else:
            R_p = update_by(p_t_old, y_t, Y, R_p, R_p_hat, R_lambda, copies, omega, omega_0, cnt_V, kappa, approx)
            R_p = update_px(X, R_p, R_p_hat, R_lambda, copies, omega, cnt_V, kappa, approx)
            # for e in Y:
            #     for k, j in copies[e]:
            #         R_p[k][j+1] = R_p_temp[k][j+1]

        p_t, b_t = R_p_2_p(R_p, copies, cnt_V)


        # print "p_t",sum(p_t)
        # print "R_p_hat", R_p_hat
        # print "R_lambda:", R_lambda
        error = np.sqrt(np.sum([np.power(p_t_old[v] - p_t[v], 2) for v in range(cnt_V)]))
        if error < epsilon:
            # print(error,len(Y),cnt_V)
            break
        sys.stdout.write("Iter-" + str(iter) + ": " + str(round(time.time() - t3, 2))+"/"+str( round(t_2 - t_1,2)) + " | ")
    sys.stdout.write("\n")
    p_adv=p_t[v0]

    # sys.exit()
    return p_adv


"""
The minimization problem to be solved is:
    min weight * c * zxy[k] + 1/(2*kappa( || zxy[k] - pxy[k] + kappa * lambda_xy[k] ||_2^2

The gradient of the objective function over zxy[k] is:
    weight * c + (1/kappa) (zxy[k] - pxy[k] + kappa * lambda_xy[k]) = 0

The updates zxy[k] can be identified such that the above gradient equals to 0:
    zxy[k] = pxy[k] - (kappa * lambda_xy[k] + weight * kappa .c)
"""

"""
INPUT
rule: a list of 4 local variables of a specific rule from one the possible types: x->x, y->x, y->y, x->y.
y_edge: a list of 2 local variables, telling the type of variable in the head and the type of variable in the tail.
        y_edge[0] = 1 indicates that the head variable is y; otherwise x.
        y_edge[1] = 1 indicates that the tail variable is y; otherwise x.

OUTPUT
distance from satisfaction of this rule
"""


def lk(rule, y_edge):
    if y_edge[0] == 1 and y_edge[1] == 1:  # both are y edges, y->y
        # y -> y
        # rule[0] and (1-rule[1]) and (1-rule[3]) -> rule[2]
        # DFS: 1 - rule[2] - (1- rule[0]) - rule[1] - rule[3] = rule[0] - rule[1] - rule[2] - rule[3]
        return rule[0] - rule[1] - rule[2] - rule[3]
    elif y_edge[0] == 0 and y_edge[1] == 1:  # x->y
        # x->y
        # rule[0] and (1-rule[3]) -> rule[2]
        # DFS: 1 - rule[2] - (1- rule[0]) - rule[3] = rule[0] - rule[2] - rule[3]
        return rule[0] - rule[2] - rule[3]
    elif y_edge[0] == 0 and y_edge[1] == 0:  # x->x
        # x -> x
        # rule[0] -> rule[2]
        # DFS: 1 - rule[2] - (1 - rule[0]) = rule[0] - rule[2]
        return rule[0] - rule[2]
    elif y_edge[0] == 1 and y_edge[1] == 0:  # y->x
        # y -> x
        # rule[0] and (1 - rule[1]) -> rule[2]
        # DFS: 1 - rule[2] - (1 - rule[0]) - rule[1] -> rule[0] - rule[1] - rule[2]
        return rule[0] - rule[1] - rule[2]


"""
INPUT
rule: a list of 4 local variables of a specific rule from one the possible types: x->x, y->x, y->y, x->y.
y_edge: a list of 2 local variables, telling the type of variable in the head and the type of variable in the tail.
        y_edge[0] = 1 indicates that the head variable is y; otherwise x.
        y_edge[1] = 1 indicates that the tail variable is y; otherwise x.

OUTPUT
c: a list of four numbers that are used to calculate the distance from satisfaction of this rule: c^T rule
"""


def delta_lk(rule, y_edge):
    if y_edge[0] == 1 and y_edge[1] == 1:  # both are y edges, y->y
        return [1.0, -1.0, -1.0, -1.0]
    elif y_edge[0] == 0 and y_edge[1] == 1:  # x->y
        return [1.0, 0.0, -1.0, -1.0]
    elif y_edge[0] == 0 and y_edge[1] == 0:  # x->x
        return [1.0, 0.0, -1.0, 0.0]
    elif y_edge[0] == 1 and y_edge[1] == 0:  # y->x
        return [1.0, -1.0, -1.0, 0.0]


"""
input rule: l_k(P)=constant+c.P
output: return constant and c
"""


def getConstantVector(rule, y_edge):
    if y_edge[0] == 1 and y_edge[1] == 1:  # both are y edges, y->y
        return 0.0, [1.0, -1.0, -1.0, -1.0]
    elif y_edge[0] == 0 and y_edge[1] == 1:  # x->y
        return 0.0, [1.0, 0.0, -1.0, -1.0]
    elif y_edge[0] == 0 and y_edge[1] == 0:  # x->x
        return 0.0, [1.0, 0.0, -1.0, 0.0]
    elif y_edge[0] == 1 and y_edge[1] == 0:  # y->x
        return 0.0, [1.0, -1.0, -1.0, 0.0]


"""
INPUT
x:
rule:
y_edge:

OUTPUT

project the solution to l_k(p,by)=0 hyperplane

Proj_{l_k=0}(x):= argmin_y ||x-y||_2   s.t. l_k(y)=0
"""


def Proj_lk(x, rule, y_edge):
    const, c = getConstantVector(rule, y_edge)
    c_part = []
    x_part = []
    idx_list = []

    if y_edge[0] == 0:  # p_x
        idx_list.append(0)
        c_part.append(c[0])
        x_part.append(x[0])
    else:  # b_y
        idx_list.append(1)
        c_part.append(c[1])
        x_part.append(x[1])
        const += c[0] * rule[0]
    if y_edge[1] == 0:  # -> p_x
        idx_list.append(2)
        c_part.append(c[2])
        x_part.append(x[2])
    else:  # ->b_y
        idx_list.append(3)
        c_part.append(c[3])
        x_part.append(x[3])
        const += c[2] * rule[2]

    c_part = np.array(c_part)
    x_part = np.array(x_part)

    y = cvx.Variable(len(c_part))
    objective = cvx.Minimize(cvx.sum_squares(x_part - y))
    constraints = [const + c_part.T * y == 0.0, y >= 0.0, y <= 1.0]
    # constraints = [const + c_part.T * y == 0.0]
    p = cvx.Problem(objective, constraints)
    p.solve()  # The optimal objective is returned by p.solve().
    # print y.value.tolist()
    y_optimal = [i for i in y.value.tolist()]
    rule_optimal = copy.deepcopy(rule)
    for i, idx in enumerate(idx_list):
        rule_optimal[idx] = y_optimal[i]
    return rule_optimal


def Proj_lk2(x, rule, y_edge):
    const, c = getConstantVector(rule, y_edge)
    c_part = []
    x_part = []
    idx_list = []
    if y_edge[0] == 0:  # p_x
        idx_list.append(0)
        c_part.append(c[0])
        x_part.append(x[0])
    else:  # b_y
        idx_list.append(1)
        c_part.append(c[1])
        x_part.append(x[1])
        const += c[0] * rule[0]
    if y_edge[1] == 0:  # -> p_x
        idx_list.append(2)
        c_part.append(c[2])
        x_part.append(x[2])
    else:  # ->b_y
        idx_list.append(3)
        c_part.append(c[3])
        x_part.append(x[3])
        const += c[2] * rule[2]
    c_part = np.array(c_part)
    x_part = np.array(x_part)

    c_norm = sqrt(list_inn(c_part, c_part))
    c_x = list_inn(c_part, x_part)
    if (c_norm == 0.0) or (c_x == 0.0):
        y_optimal = x_part
    else:
        y_optimal = list_minus(x_part, list_dot_scalar(c_part, c_x / c_norm))

    rule_optimal = copy.deepcopy(rule)
    for i, idx in enumerate(idx_list):
        rule_optimal[idx] = y_optimal[i]
    return rule_optimal
# for idx, type in enumerate(y_edge):
#     if type == 0 and proj[idx * 2+1] is not 0:
#         warnings.warn("b_xi = 0 in the function Proj_lk: {}".format(proj[idx * 2+1]))

"""
INPUT
R_p_b:
    R_p_b[k][j * 2] refers to the probability variable of j-th edge in the k-th rule in R. Note that the are multiple copies of each sample probability value (p_xi or p_yi)
    R_p_b[k][j * 2+1] relates to the j-th edge in the k-th rule in R and is a binary value indicating if this edge has been compromised.
    Note that each edge has a probability value and also a bollean value
copies: copies[e] is a list indexes of the variables in the set of rules R that are related to the edge e.
        for k, j in copies[e]:
            k is the index of a specific rule and j is index of a specific edge in the rule.
cnt_V: total number of edges in the network

OUTPUT
p: a dictionary of probability values: key is an edge id and value is the probability value of this edge
b: a dictionary of probability values: key is an edge id and value is the binary value of this edge indicating if this edge is compromised or not.
"""


def R_p_2_p(R_p, copies, cnt_V):
    b = [-1.0 for i in range(cnt_V)]
    p = [-1.0 for i in range(cnt_V)]
    for v, v_copies in copies.items():
        # k, j = v_copies[0]  # k_th rule, j_th item
        # p[v] = np.round(R_p[k][j], 2)
        # b[v] = np.round(R_p[k][j + 1], 2)
        """p_x,p_y and b_y are mean of their local copies """
        p_v = []
        b_v = []
        for k, j in v_copies:  # k_th rule, j_th item
            p_v.append(R_p[k][j])
            b_v.append(R_p[k][j+1])
        # p[v] = round(np.round(np.mean(p_v),2))
        # b[v] = round(np.round(np.mean(b_v),2))
        if p[v]>1.0 or b[v]>1.0 : print("?????????????????????????????????????????????????")
        p[v] = np.round(np.mean(p_v), 2)
        b[v] = np.round(np.mean(b_v), 2)

    return p, b


def rround(myList):
    myFormattedList = [round(elem, 3) for elem in myList]
    return myFormattedList


def rround2(myListAll):
    myFormattedList = []
    for myList in myListAll:
        myFormattedList.append([round(elem, 3) % elem for elem in myList])
    return myFormattedList

def gen_nns(V, E):
    nns = {v: [] for v in V}
    for (v1,v2) in E:
        if nns.has_key(v1):
            nns[v1].append(v2)
    return nns

def reformat_nodes_in_omega(Omega, node_2_id):
    Omega1 = {}
    for v, alpha_beta in Omega.items():
        Omega1[node_2_id[v]] = alpha_beta
    return Omega1

"""
INPUT
V = [0, 1, ...] is a list of vertex ids
E: a list of pairs of vertex ids
Obs: a dictionary with key edge and its value a list of congestion observations of this edge from t = 1 to t = T
Omega: a dictionary with key edge and its value a pair of alpha and beta

OUTPUT
V: same as input
edge_up_nns: key is an edge id, and its value is a list of its up stream neighbor edge ids.
edge_down_nns: key is an edge id, and its value is a list of its down stream neighbor edge ids.
edge_2_id: a dictionary with key a directed edge and its value is the id of this edge.
id_2_edge: a dictionary with key an edge id and its value the corresponding edge
omega: A dictionary of opinions (tuples of alpha and beta values): [edge1 id: [alpha_1, beta_1], ..., ]. The size of the dictionary is the total number of edges.
feat: A dictionary of lists of observations: [edge1 id: [y_1^1, ..., y_1^T], ...].
"""


def reformat_nodes(V, E, Obs, Omega = None):
    feat = {}
    id_2_node = {}
    node_2_id = {}
    for idx, v in enumerate(V.keys()):
        id_2_node[idx] = v
        node_2_id[v] = idx
    for v, v_feat in Obs.items():
        feat[node_2_id[v]] = v_feat
    if Omega != None:
        Omega1 = reformat_nodes_in_omega(Omega, node_2_id)
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


    return node_nns, id_2_node, node_2_id, Omega1, feat


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


def calculate_measures(true_omega_x, pred_omega_x, b, pred_b_y, X, logging):
    W = 1.0
    a = 0.5
    """
    Calculate b_f_measure
    """
    cnt_Y = len(pred_b_y)
    n_matches = 0.0
    n_positives = 0.0
    n_total = 0.0
    for e, val in pred_b_y.items():
        if b[e] == val and val == 1:
            n_matches += 1
        if b[e] == 1:
            n_positives += 1
        if val == 1:
            n_total += 1
    if n_total == 0: n_total = 1
    b_prec = n_matches / n_total
    b_recall = n_matches / n_positives
    if (b_prec + b_recall) > 0:
        b_f_measure = 2 * (b_prec * b_recall) / (b_prec + b_recall)
    else:
        b_f_measure = 0

    """
    Calculate the quality measures of uncertainty prediction
    """
    u_true_X = {e: np.abs((W * 1.0) / (true_omega_x[e][0] + true_omega_x[e][1])) for e in X}
    u_pred_X = {e: np.abs((W * 1.0) / (pred_omega_x[e][0] + pred_omega_x[e][1])) for e in X}
    # print "True Omega:",true_omega_x
    # print "pred Omega:",pred_omega_x
    u_mse = np.mean([np.abs(u_pred_X[e] - u_true_X[e]) for e in X])
    u_relative_mse = np.mean([np.abs(u_pred_X[e] - u_true_X[e]) / u_true_X[e] for e in X])

    """
    EB-MSE= 1/N \sum |a_i/(a_i+b_i) - a*_i/(a*_i+b*_i)|
    """
    prob_true_X = {e: (true_omega_x[e][0] * 1.0) / (true_omega_x[e][0] + true_omega_x[e][1]) + 0.0001 for e in X}
    prob_pred_X = {e: (pred_omega_x[e][0] * 1.0) / (pred_omega_x[e][0] + pred_omega_x[e][1]) for e in X}

    prob_mse = np.mean([np.abs(prob_pred_X[e] - prob_true_X[e]) for e in X])
    prob_relative_mse = np.mean([np.abs(prob_pred_X[e] - prob_true_X[e]) / prob_true_X[e] for e in X])

    # logging.write("********************* uncertainty MSE: {0}, Relative-MSE: {1}".format(u_mse, u_relative_mse))
    # logging.write("********************* probability MSE: {0}, Relative-RMSE: {1}".format(prob_mse, prob_relative_mse))

    recall_congested = 0.0
    n_congested = 0.01
    recall_uncongested = 0.0
    n_uncongested = 0.01
    for e in X:
        if prob_true_X[e] >= 0.5:
            n_congested += 1
            if prob_pred_X[e] > 0.5:
                recall_congested += 1
        else:
            n_uncongested += 1
            if prob_pred_X[e] < 0.5:
                recall_uncongested += 1
    accuracy = (recall_congested + recall_uncongested) * 1.0 / (n_congested + n_uncongested)
    if recall_congested > 0:
        recall_congested = recall_congested / n_congested
    else:
        # print "recall congeste",recall_congested
        recall_congested = -1
    if recall_uncongested > 0:
        recall_uncongested = recall_uncongested / n_uncongested
    else:
        recall_uncongested = -1
    # logging.write("********************* accuracy: {0}, recall congested: {1}, recall uncongested: {2}".format(accuracy, recall_congested, recall_uncongested))
    return prob_mse, u_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested, b_f_measure


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


def evaluate(V, E, Obs, Omega, X, Y, b, logging, psl=False, approx=False, init_alpha_beta=(1, 1), report_stat=False):
    pred_omega_x, pred_b_y = inference_apdm_format(V, E, Obs, Omega, b, X, logging, psl, approx, init_alpha_beta,
                                                   report_stat)
    prob_mse, u_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested, b_f_measure = calculate_measures(
        Omega, pred_omega_x, b, pred_b_y, X, logging)
    str = "(true bi, predicted bi): "
    cnt = 1
    for e, val in b.items():
        if e in Y:
            if cnt > 1:
                str += ", "
            if int(val) is not int(pred_b_y[e]):
                str += color.BOLD + color.RED + "{}: ({}, {})".format(e, val, pred_b_y[e]) + color.END + color.END
            else:
                str += "{}: ({}, {})".format(e, val, pred_b_y[e])
        cnt += 1
    logging.write(str)
    str = ""
    if b_f_measure < 0.9:
        str += color.RED + "b_F_measure: {}".format(b_f_measure) + color.END
    else:
        str += "b_F_measure: " + color.BOLD + "{}".format(b_f_measure) + color.END
    str += "; "
    if prob_mse > 0.1:
        str += color.RED + "omega_projection_prob_mse: {}".format(prob_mse) + color.END
    else:
        str += "omega_projection_prob_mse: " + color.BOLD + "{}".format(prob_mse) + color.END
    logging.write(str)
    logging.write("")  # add an empty line
    return prob_mse, u_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested, b_f_measure


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


def generate_a_random_opinion(a=0.5, W=1):
    W = 1
    u = random.uniform(0.01, 0.5)
    b = random.uniform(0.01, 1 - u)
    d = 1 - u - b
    r = b * W / u
    s = d * W / u
    alpha = r + a * W
    beta = s + (1 - a) * W
    return (alpha, beta)


"""
INPUT
V: a list of vertex ids
E: a list of ordered pairs of vertex ids
T: The number of observations. 100 by default.

OUTPUT
datasets: key is rate and the value is [[V, E, Omega, Obs, X], ...], where rate refers to the rate of edges that are randomly selected as target edges

"""


def simulation_data_generator(V, E, rates=[0.05, 0.1, 0.15, 0.25, 0.3, 0.4, 0.5], realizations=10, T=50):
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




def main():
    return
    # testcase0()
    # testcase0()
    # testcase1()
    # simulation_data_generator1()
    # sampple_epinion_network()

    # print np.arange(0, 1, 0.01)
    # testcase2()
    # E_X = [(2, 7), (7, 12), (15, 16), (18, 19)]
    # E_X = [(2, 7), (7, 12)]
    # for i in range(1, 101):
    #     filename = "APDM-GirdData-k{0}.txt".format(i)
    #     print "-----------------------{0}".format(filename)
    #     V, E, Obs, Omega = readAPDM(filename)
    #     evaluate(V, E, Obs, Omega, E_X)
    #     # break


if __name__ == '__main__':
    main()

