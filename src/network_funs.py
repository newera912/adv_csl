__author__ = 'fengchen'
from basic_funs import *

import numpy as np

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


"""
INPUT
a: base with default 0.5
W:

OUTPUT
a randomly generated opinion (alpha, beta)
"""
def generate_a_random_opinion(a = 0.5, W = 2):
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
V = [0, 1, ...] is a list of vertex ids
E: a list of pairs of vertex ids
Obs: a dictionary with key edge and its value a list of congestion observations of this edge from t = 1 to t = T
Omega: a dictionary with key edge and its value a pair of alpha and beta
E_X: a list of pairs of vertex ids whose opinions will be predicted.

OUTPUT
omega_X: {edge: (alpha, beta), ...}, where edge \in E_X and is represented by a pair of vertex ids
"""
def SL_prediction1(V, E, Omega, E_X):
    omega_X = {e: alpha_beta for e, alpha_beta in Omega.items() if e in E_X}
    return omega_X


def gen_nns(V, E):
    nns = {v: [] for v in V}
    for (v1,v2) in E:
        if nns.has_key(v1):
            nns[v1].append(v2)
    return nns

"""
INPUT
V = [0, 1, ...] is a list of vertex ids
E: a list of pairs of vertex ids
Obs: a dictionary with key edge and its value a list of congestion observations of this edge from t = 1 to t = T
Omega: a dictionary with key edge and its value a pair of alpha and beta

OUTPUT
edge_up_nns: key is an edge id, and its value is a list of its up stream neighbor edge ids.
edge_down_nns: key is an edge id, and its value is a list of its down stream neighbor edge ids.
edge_2_id: a dictionary with key a directed edge and its value is the id of this edge.
id_2_edge: a dictionary with key an edge id and its value the corresponding edge
Omega1: A dictionary of opinions (tuples of alpha and beta values): [edge1 id: [alpha_1, beta_1], ..., ]. The size of the dictionary is the total number of edges.
feat: A dictionary of lists of observations: [edge1 id: [y_1^1, ..., y_1^T], ...].
"""
def reformat(V, E, Obs, Omega = None):
    feat = {}
    id_2_edge = {}
    edge_2_id = {}
    for idx, e in enumerate(E):
        id_2_edge[idx] = e
        edge_2_id[e] = idx
    for e, e_feat in Obs.items():
        feat[edge_2_id[e]] = e_feat
    if Omega != None:
        Omega1 = reformat_edge_in_omega(Omega, edge_2_id)
    else:
        Omega1 = None
    # for e, alpha_beta in Omega.items():
    #     omega[edge_2_id[e]] = alpha_beta
    edges_end_at = {}
    edges_start_at = {}
    for e in E:
        e_id = edge_2_id[e]
        start_v = e[0]
        end_v = e[1]
        if edges_end_at.has_key(end_v):
            edges_end_at[end_v][e_id] = 1
        else:
            edges_end_at[end_v] = {e_id: 1}
        if edges_start_at.has_key(start_v):
            edges_start_at[start_v][e_id] = 1
        else:
            edges_start_at[start_v] = {e_id: 1}
    edge_up_nns = {}
    edge_down_nns = {}
    for e in E:
        start_v = e[0]
        end_v = e[1]
        e_id = edge_2_id[e]
        if edges_end_at.has_key(start_v):
            edge_up_nns[e_id] = edges_end_at[start_v]
        if edges_start_at.has_key(end_v):
            edge_down_nns[e_id] = edges_start_at[end_v]
    return edge_up_nns, edge_down_nns, id_2_edge, edge_2_id, Omega1, feat

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


    return node_nns, id_2_node, node_2_id, Omega1, feat

def reformat_edge_in_omega(Omega, edge_2_id):
    Omega1 = {}
    for e, alpha_beta in Omega.items():
        Omega1[edge_2_id[e]] = alpha_beta
    return Omega1

def get_edge_nns(V, E):
    edges_end_at = {}
    edges_start_at = {}
    for e in E:
        start_v = e[0]
        end_v = e[1]
        if edges_end_at.has_key(end_v):
            edges_end_at[end_v][e] = 1
        else:
            edges_end_at[end_v] = {e: 1}
        if edges_start_at.has_key(start_v):
            edges_start_at[start_v][e] = 1
        else:
            edges_start_at[start_v] = {e: 1}
    edge_up_nns = {}
    edge_down_nns = {}
    for e in E:
        start_v = e[0]
        end_v = e[1]
        if edges_end_at.has_key(start_v):
            edge_up_nns[e] = edges_end_at[start_v]
        if edges_start_at.has_key(end_v):
            edge_down_nns[e] = edges_start_at[end_v]
    return edge_up_nns, edge_down_nns

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



# def calc_Omega_from_Obs(Obs, E):
#     T = len(Obs.values()[0])
#     Omega = {}
#     for e in E:
#         Omega[e] = (np.sum(Obs[e]) * 1.0, T - np.sum(Obs[e]) * 1.0)
#     return Omega

#Omega[e] = (np.sum(Obs[e]) * 0.5, T - np.sum(Obs[e]) * 1.0+0.5)

def calc_Omega_from_Obs2(Obs, E):
    W=2.0
    a=0.5
    T = len(Obs.values()[0])
    Omega = {}
    for e in E:
        pos_evidence=np.sum(Obs[e])* 1.0
        neg_evidence=T - pos_evidence
        b = pos_evidence/(pos_evidence+neg_evidence+W)
        d = neg_evidence / (pos_evidence + neg_evidence + W)
        u = W/(pos_evidence + neg_evidence + W)
        alpha=W*b/u + W*a
        beta= W*d/u + W*(1-a)
        Omega[e] = (alpha,beta)
    return Omega

# def calc_Omega_from_Obs_nodes(Obs, V):
#     W=2.0
#     a=0.5
#     T = len(Obs.values()[0])
#     Omega = {}
#     for v in V:
#         pos_evidence=np.sum(Obs[v])* 1.0
#         neg_evidence=T - pos_evidence
#         b = pos_evidence/(pos_evidence+neg_evidence+W)
#         d = neg_evidence / (pos_evidence + neg_evidence + W)
#         u = W/(pos_evidence + neg_evidence + W)
#         alpha=W*b/u + W*a
#         beta= W*d/u + W*(1-a)
#         Omega[v] = (alpha,beta)
#     return Omega

def calc_Omega_from_Obs(Obs, E):
    T = len(Obs.values()[0])
    Omega = {}
    for e in E:
        Omega[e] = (np.sum(Obs[e]) * 0.5, T - np.sum(Obs[e]) * 1.0 + 0.5)
    return Omega
"""
INPUT
p: [p_1, p_2, ... p_T]. p_t is a list of truth probability values of all edges. p[t][i] refers to the probability value of edge id=i at time t, where t \in {1, ..., T} and i \in {1, cnt_Y+cnt_X}
omega: A dictionary of opinions (tuples of alpha and beta values) of all edges (X+Y): [edge1 id: [alpha_1, beta_1], ...]. Note that the opinions related to the edges in X will be udpated based on p.
X: a list of edge ids whose opinions need to be predicted

OUTPUT
omega_x: A dictionary of opinions (tuples of alpha and beta values): [edge1 id: [alpha_1, beta_1], ..., edge_N id: [alpha_M, beta_N]]. cnt_X is the size of X.
"""
def estimate_omega_x(p, omega, X):
    # print "p", p
    # print "omega", omega
    # print "X", X

    strategy = 1 # 1: means we consider p values as binary observations and use them to estimate alpha and beta.
    if strategy == 1:
        for e in X:
            data = [p_t[e] for p_t in p]
            """Original method"""
            # alpha1 = np.sum(data)
            # beta1 = np.sum([1 - prob for prob in data])
            """New methods"""
            alpha1 = np.sum(data) + 0.5
            beta1 = len(data) - np.sum(data) + 0.5
            # print "data, alpha1, beta1", data, alpha1, beta1
            omega[e] = (alpha1, beta1)
    else:
        for e in X:
            data = [p_t[e] for p_t in p]
            if np.std(data) < 0.01:
                alpha1 = np.mean(data)
                beta1 = 1 - alpha1
            else:
                alpha1, beta1, loc, scale = beta.fit(data, floc=0,fscale=1)
            alpha1 = alpha1 / (alpha1 + beta1) * 10
            beta1 = 10 - alpha1
            omega[e] = (alpha1, beta1)
    return omega

def estimate_omega_x(ps, X):
    omega_x = {}
    strategy = 1 # 1: means we consider p values as binary observations and use them to estimate alpha and beta.
    if strategy == 1:
        for e in X:
            data = [prob_2_binary(p_t[e]) for p_t in ps]
            # if sum(data)!=0.0: print sum(data),data
            # data = [(p_t[e]) for p_t in ps]
            alpha1 = np.sum(data) + 0.5
            beta1 = len(data) - np.sum(data) + 0.5
            omega_x[e] = (alpha1, beta1)
    return omega_x


def prob_2_binary(val):
    return val
    # if val > 0.45:
    #     return 1.0
    # else:
    #     return 0.0


def prob_2_binary2(val):
    # return val
    if val >= 0.2:
        return 1
    else:
        return 0

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
edge_up_nns: key is an edge id, and its values is a list of its downstream and neighboring edge ids. Here up is defined based on the interpretation of the path as a tree.

OUTPUT
R: a list of pairs of neighboring edge ids.
"""
def generate_PSL_rules_from_edge_cnns(edge_up_nns):
    dic_R = {}
    for e, neighbors in edge_up_nns.items():
        for up_e in neighbors.keys():
            if not dic_R.has_key((e, up_e)):  # The direction should be from up_e to e. The traffic at e will impact the traffic of its up adjacent edges
                dic_R[(e, up_e)] = 1
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
def calc_initial_p(y_t, edge_down_nns, X, Y, cnt_E, p0):
    p = [0 for i in range(cnt_E)]
    for e, e_nns in edge_down_nns.items():
        if X.has_key(e):
            obs = [y_t[e_n] for e_n in e_nns.keys() if Y.has_key(e_n)]
            if len(obs) == 0: obs = [p0]
            # print "obs:", e, obs
            p[e] = np.median(obs)
        else:
            p[e] = y_t[e]
    return p

# def calc_initial_p(y_t, edge_down_nns, X, Y, cnt_E, p0):
#     p = [0 for i in range(cnt_E)]
#     for e in Y:
#         p[e] = y_t[e]
#
#     for e, e_nns in edge_down_nns.items():
#         if X.has_key(e):
#             obs = [y_t[e_n] for e_n in e_nns.keys() if Y.has_key(e_n)]
#             if len(obs) > 0:
#                 p[e] = np.median(obs)
#         else:
#             p[e] = y_t[e]
#
#     for rep in range(3):
#         for e in X:
#             if
#         for e, e_nns in edge_down_nns.items():
#             if X.has_key(e):
#                 obs = [y_t[e_n] for e_n in e_nns.keys() if Y.has_key(e_n)]
#                 if len(obs) > 0:
#                     p[e] = np.median(obs)
#
#     # for e, e_nns in edge_down_nns.items():
#     #     if X.has_key(e):
#     #         obs = [y_t[e_n] for e_n in e_nns.keys() if Y.has_key(e_n)]
#     #         if len(obs) == 0: obs = [p0]
#     #         # print "obs:", e, obs
#     #         p[e] = np.median(obs)
#     #     else:
#     #         p[e] = y_t[e]
#     return p

def calc_initial_p1(y_t, edge_down_nns, X, Y, cnt_E, p0):
    p = [0 for i in range(cnt_E)]
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
            # else:
            #     if n_pos==0:
            #         p[e] = 0.0
            #     else:
            #         p[e] = 1.0
            # p[e] = np.median(obs)
        else:
            p[e] = y_t[e]
    return p

def calc_initial_p1_nodes(y_t, node_nns, X, Y, cnt_V, p0):
    p = [0 for i in range(cnt_V)]
    for v, v_nns in node_nns.items():
        if X.has_key(v):
            obs = {v_n:y_t[v_n] for v_n in v_nns if Y.has_key(v_n)}
            # if len(obs) == 0: obs = {p0}
            conf = 0
            n_pos=0.0
            for v_o,val in obs.items():
                n_pos+=val
                if val > 0:
                    conf += 1.0
                else:
                    conf -= 1.0
            if conf > 0:
                p[v] = 1.0
            # else:
            #     if n_pos==0:
            #         p[e] = 0.0
            #     else:
            #         p[e] = 1.0
            # p[e] = np.median(obs)
        else:
            p[v] = y_t[v]
    return p

def calc_initial_p_1(y_t, edge_nns, Y, cnt_E, p0):
    p = [0 for i in range(cnt_E)]
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
        if v[i] < 0: v[i] = 0
        if v[i] > 1: v[i] = 1
    return v



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
    # for prob in np.arange(0.01, 1, 0.01):
    probs=list(np.arange(0.1, 1, 0.1))
    probs.extend([0.01,0.99])
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
        if e in Y: continue
        nc = len(copies[e]) * 1.0
        z_lambda_sum = sum([R_z[k][j] + R_lambda_[k][j] / rho for k, j in copies[e]])
        prob = z_lambda_sum / nc
        if prob < 0: prob = 0
        if prob > 1: prob = 1
        for k, j in copies[e]:
            R_p[k][j] = prob
    return R_p




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

