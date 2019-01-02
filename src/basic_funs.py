__author__ = 'fengchen'
from cubic import cubic
from math import *
import numpy as np

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

def trim(val):
    if val < 0:
        return 0
    elif val > 1:
        return 1
    else:
        return val 

def beta_to_opinion2(alpha,beta,W=2.0,a=0.5):
    '''
    compute opinion based on hyperparameters of beta distribution
    '''
    r = alpha - W *a
    s= beta - W*(1-a)
    b = trim((r)/float(r+s+W))
    d = trim((s)/float(r+s+W))
    u = trim(W/float(r+s+W))
    return [b,d,u,a]

def beta_to_opinion(alpha,beta,W=2.0,a=0.5):
    '''
    compute opinion based on hyperparameters of beta distribution
    '''
    b = trim((alpha - W*a)/float(alpha+beta))
    d = trim((beta - W*(1-a))/float(alpha+beta))
    u = trim(W/float(alpha+beta+W))
    return [b,d,u,a]

def opinion_to_beta(op,W=2.0):
    '''
    convert opinion to hyperparameters of beta distribution
    op is opinion : list
    '''
    b,d,u,a = op
    #alpha = W*a + (b*W/u)
    #beta = (W-u*W*a-b*W)/u
    r = b * W / float(u)
    s = d * W / float(u)
    alpha = r + a * W
    if alpha < 0: alpha = 0
    beta = s + (1.0-a) * W
    if beta < 0: beta = 0
    return (alpha,beta)

def get_edge_id(v1, v2, edge2id):
    if v1 <= v2:
        return edge2id[(v1, v2)]
    else:
        return edge2id[(v2, v1)]

"""
INPUT
edge_nns: key is an edge id, and its value is a list of its neighbor edge ids.
id_2_edge:
edge_2_id:

OUTPUT
R: a list of pairs of neighboring edge ids.
"""
def generate_eopinion_PSL_rules_from_edge_cnns(edge_down_nns, id_2_edge, edge_2_id):
    dic_R = {}
    dict_paths = {}
    for a_b, down_nns in edge_down_nns.items():
        for b_c in down_nns.keys():
            if edge_2_id.has_key((id_2_edge[a_b][0], id_2_edge[b_c][1])):
                a_c = edge_2_id[(id_2_edge[a_b][0], id_2_edge[b_c][1])]
                if not dic_R.has_key((a_b, b_c, a_c)):
                    dic_R[(a_b, b_c, a_c)] = 1  # debug. Will dic_R.items() always be fixed, if it is not updated?
                    if dict_paths.has_key((a_c)):
                        dict_paths[a_c].append((a_b, b_c))
                    else:
                        dict_paths[a_c] = [(a_b, b_c)]
    return dic_R.keys(), dict_paths

def accuracy_2_posi_nega(accuracy):
    if np.min([accuracy]) > 0.9:
        (i_nposi, i_nnega) = (1, 0)
    else:
        (i_nposi, i_nnega) = (0, 1)
    return i_nposi, i_nnega

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
