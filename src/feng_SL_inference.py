__author__ = 'Chunpai W.'

# import networkx as nx
import numpy as np
import pdb
import copy
from basic_funs import *
# from readAPDM import readAPDM
import random,time
# from SLOperators import transitivity
# from SLOperators import fusion
from random import randint
from random import shuffle
# from DataWrapper import *
from random import randint

threshold=10
# global paths
paths = []
# global path
path = []


def findPaths(nns, e):
    # if len(paths) > 5: return
    # global paths, path
    # print source, target, threshold
    arcs = []
    source, target = e
    for nei in nns[source]:
        # for nei in nx.neighbors(G,source):
        arc = (source, nei)
        length = len(path)
        # path_copy = copy.copy(path)
        count = 0
        for (i, j) in path:
            if nei != i and nei != j:
                count += 1
        if count == length:
            arcs.append(arc)
    for arc in arcs:
        # print path
        if arc[1] == target and len(path) + 1 <= threshold:
            path_copy = copy.copy(path)
            path_copy.append(arc)
            paths.append(path_copy)
            # if len(paths)>5: return
        elif arc[1] != target and len(path) + 1 <= threshold:
            # if len(paths) > 5: return
            path.append(arc)
            findPaths(nns, (arc[1], target))
            path.remove(arc)


def get_v_edge_nns(V, E):
    nns = {}
    edges_end_at = {}
    edges_start_at = {}
    for e in E:
        start_v = e[0]
        end_v = e[1]
        if nns.has_key(start_v):
            nns[start_v].append(end_v)
        else:
            nns[start_v] = [end_v]
        if not nns.has_key(end_v): nns[end_v] = []

        if edges_end_at.has_key(end_v):
            edges_end_at[end_v].append(e)
        else:
            edges_end_at[end_v] = [e]
        if not edges_end_at.has_key(start_v):
            edges_end_at[start_v] = []
        if edges_start_at.has_key(start_v):
            edges_start_at[start_v].append(e)
        else:
            edges_start_at[start_v] = [e]
        if not edges_start_at.has_key(end_v):
            edges_start_at[end_v] = []
    edge_up_nns = {}
    edge_down_nns = {}
    for e in E:
        start_v = e[0]
        end_v = e[1]
        if edges_end_at.has_key(start_v):
            edge_up_nns[e] = edges_end_at[start_v]
        else:
            edge_up_nns[e] = []
        if edges_start_at.has_key(end_v):
            edge_down_nns[e] = edges_start_at[end_v]
        else:
            edge_down_nns[e] = []
    return edges_start_at, edges_end_at, nns


def update(edges_start_at, edges_end_at, nns, e):
    if edges_start_at.has_key(e[0]):
        edges_start_at[e[0]].append(e)
    else:
        edges_start_at[e[0]] = [e]
    if not edges_start_at.has_key(e[1]):
        edges_start_at[e[1]] = []

    if edges_end_at.has_key(e[1]):
        edges_end_at[e[1]].append(e)
    else:
        edges_end_at[e[1]] = [e]
    if not edges_end_at.has_key(e[0]):
        edges_end_at[e[0]] = []

    if nns.has_key(e[0]):
        nns[e[0]].append(e[1])
    else:
        nns[e[0]] = [e[1]]
    if not nns.has_key(e[1]): nns[e[1]] = []

    return edges_start_at, edges_end_at, nns


def SL_prediction(V, E, Obs, Omega, E_X, flag=0):
    # print "Path len:",threshold
    global paths, path
    T=len(Obs[E[0]])
    #when alpha=3.0, beta=3.0, then the u=1/3,d=1/3,u=1/3
    alpha0=1.0
    beta0=1.0
    # E_X1 = copy.deepcopy(E_X)
    E_X1 = {e:1 for e in E_X}
    E_op = {}
    V_checked = {}
    for e in E:
        if not E_X1.has_key(e):
            alpha, beta = Omega[e]
            E_op[e] = beta_to_opinion2(alpha, beta)
            V_checked[e[0]] = 1
            V_checked[e[1]] = 1
    # print V_checked.keys(), E_op.keys()
    edges_start_at, edges_end_at, nns = get_v_edge_nns(V_checked.keys(), E_op.keys())
    Omega_X = {}
    if flag == 0:
        # print 'flag == 0'
        while (len(E_X1) != 0):
            # X = [e for e in E_X1]
            X = E_X1.keys()
            # shuffle(X)
            t0=time.time()
            t3 = time.time()
            i=0
            for e in X:
                # if len(E_X1)%1000==0:
                #     print len(E_X1),t0-t3,"****************************************************************************"
                #     t0=time.time()
                source, target = e
                # case 1
                if not V_checked.has_key(source) and not V_checked.has_key(target):
                    # print e
                    continue

                #case 2
                elif not V_checked.has_key(source):
                    V_checked[source] = 1
                    neighbor_path = []
                    for e_nn in edges_start_at[e[1]]:
                        neighbor_path.append(e_nn)

                    multi_paths = []
                    if neighbor_path == []:
                        Omega_X[e] = (alpha0,beta0)
                        # alpha_u=randint(1, T)
                        # beta_u=randint(1, T)
                        # Omega_X[e] = (alpha_u, beta_u)
                        # opi = beta_to_opinion(1.0,1.0)
                        E_op[e] = beta_to_opinion2(alpha0,beta0)
                        edges_start_at, edges_end_at, nns = update(edges_start_at, edges_end_at, nns, e)
                        # raise Exception('dead edge encounted.')
                    else:
                        multi_paths.append(neighbor_path)
                        op = computeOpinion(E_op, e, multi_paths)
                        E_op[e] = op
                        edges_start_at, edges_end_at, nns = update(edges_start_at, edges_end_at, nns, e)
                        # G.add_edge(source,target,opinion = op)
                        Omega_X[e] = opinion_to_beta(op)
                    # E_X1.remove(e)
                    E_X1.pop(e, None)
                #case 3
                elif not V_checked.has_key(target):
                    V_checked[target] = 1
                    multi_paths = []
                    neighbor_path = []
                    for e_nn in edges_end_at[e[0]]:
                        neighbor_path.append(e_nn)

                    if neighbor_path == []:
                        Omega_X[e] = (alpha0,beta0)
                        # alpha_u = randint(1, T)
                        # beta_u = randint(1, T)
                        # Omega_X[e] = (alpha_u, beta_u)
                        E_op[e] = beta_to_opinion2(alpha0,beta0)
                        edges_start_at, edges_end_at, nns = update(edges_start_at, edges_end_at, nns, e)
                    else:
                        multi_paths.append(neighbor_path)
                        op = computeOpinion(E_op, e, multi_paths)
                        E_op[e] = op
                        edges_start_at, edges_end_at, nns = update(edges_start_at, edges_end_at, nns, e)
                        Omega_X[e] = opinion_to_beta(op)
                    E_X1.pop(e, None)
                #case 4
                else:
                    # print 'CASE 4-------------------------------------------'
                    # print "start---"
                    multi_paths = []

                    for e_nn in edges_end_at[e[0]] + edges_start_at[e[1]]:
                        multi_paths.append([e_nn])
                    # print "*********** begin finding path"
                    # t_0=time.time()
                    paths = []
                    path = []
                    findPaths(nns, e)
                    # print i,time.time()-t_0
                    # print "*********** end finding path"
                    multi_paths += paths
                    if multi_paths != []:
                        paths = []
                        total_opinion = computeOpinion(E_op, e, multi_paths)
                        E_op[e] = total_opinion
                        edges_start_at, edges_end_at, nns = update(edges_start_at, edges_end_at, nns, e)
                        Omega_X[e] = opinion_to_beta(total_opinion)
                        E_X1.pop(e, None)
                    else:
                        Omega_X[e] = (alpha0,beta0)
                        # alpha_u = randint(1, T)
                        # beta_u = randint(1, T)
                        # Omega_X[e] = (alpha_u, beta_u)
                        E_op[e] = beta_to_opinion2(alpha0,beta0)
                        edges_start_at, edges_end_at, nns = update(edges_start_at, edges_end_at, nns, e)
                        E_X1.pop(e, None)
                t3=time.time()
                i+=1
    elif flag == 1:
        while (len(E_X1) != 0):
            for e in E_X1.keys():
                # print 'testing edge',e
                # source, target = e
                # print "*********** begin finding path"
                paths = []
                path = []
                findPaths(nns, e)
                # print "*********** end finding path"
                # global paths
                # print e, paths
                if paths != []:
                    total_opinion = computeOpinion(E_op, e, paths)
                    E_op[e] = total_opinion
                    edges_start_at, edges_end_at, nns = update(edges_start_at, edges_end_at, nns, e)
                    # G.add_edge(source, target, opinion = total_opinion)
                    Omega_X[e] = opinion_to_beta(total_opinion)
                    paths = []  # set global variable paths to [] to ensure next function call of findPaths()
                    # E_X1.remove(e)
                    E_X1.pop(e, None)
                else:
                    raise Exception('no path found on edge:', e)
    else:
        raise Exception('wrong flag parameter')
    # print G.number_of_nodes()
    # print G.number_of_edges()
    # print "SL is Done"
    paths = []
    path = []
    return Omega_X
def SL_prediction_multicore(V, E, Obs, Omega, E_X, flag=0):
    # print "Path len:",threshold
    global paths, path
    T=len(Obs[E[0]])
    #when alpha=3.0, beta=3.0, then the u=1/3,d=1/3,u=1/3
    alpha0=1.0
    beta0=1.0
    # E_X1 = copy.deepcopy(E_X)
    E_X1 = {e:1 for e in E_X}
    E_op = {}
    V_checked = {}
    for e in E:
        if not E_X1.has_key(e):
            alpha, beta = Omega[e]
            E_op[e] = beta_to_opinion2(alpha, beta)
            V_checked[e[0]] = 1
            V_checked[e[1]] = 1
    # print V_checked.keys(), E_op.keys()
    edges_start_at, edges_end_at, nns = get_v_edge_nns(V_checked.keys(), E_op.keys())
    Omega_X = {}
    if flag == 0:
        # print 'flag == 0'
        while (len(E_X1) != 0):
            # X = [e for e in E_X1]
            X = E_X1.keys()
            # shuffle(X)
            t0=time.time()
            t3 = time.time()
            i=0
            for e in X:
                # if len(E_X1)%1000==0:
                #     print len(E_X1),t0-t3,"****************************************************************************"
                #     t0=time.time()
                source, target = e
                # case 1
                if not V_checked.has_key(source) and not V_checked.has_key(target):
                    # print e
                    continue

                #case 2
                elif not V_checked.has_key(source):
                    V_checked[source] = 1
                    neighbor_path = []
                    for e_nn in edges_start_at[e[1]]:
                        neighbor_path.append(e_nn)

                    multi_paths = []
                    if neighbor_path == []:
                        Omega_X[e] = (alpha0,beta0)
                        # alpha_u=randint(1, T)
                        # beta_u=randint(1, T)
                        # Omega_X[e] = (alpha_u, beta_u)
                        # opi = beta_to_opinion(1.0,1.0)
                        E_op[e] = beta_to_opinion2(alpha0,beta0)
                        edges_start_at, edges_end_at, nns = update(edges_start_at, edges_end_at, nns, e)
                        # raise Exception('dead edge encounted.')
                    else:
                        multi_paths.append(neighbor_path)
                        op = computeOpinion(E_op, e, multi_paths)
                        E_op[e] = op
                        edges_start_at, edges_end_at, nns = update(edges_start_at, edges_end_at, nns, e)
                        # G.add_edge(source,target,opinion = op)
                        Omega_X[e] = opinion_to_beta(op)
                    # E_X1.remove(e)
                    E_X1.pop(e, None)
                #case 3
                elif not V_checked.has_key(target):
                    V_checked[target] = 1
                    multi_paths = []
                    neighbor_path = []
                    for e_nn in edges_end_at[e[0]]:
                        neighbor_path.append(e_nn)

                    if neighbor_path == []:
                        Omega_X[e] = (alpha0,beta0)
                        # alpha_u = randint(1, T)
                        # beta_u = randint(1, T)
                        # Omega_X[e] = (alpha_u, beta_u)
                        E_op[e] = beta_to_opinion2(alpha0,beta0)
                        edges_start_at, edges_end_at, nns = update(edges_start_at, edges_end_at, nns, e)
                    else:
                        multi_paths.append(neighbor_path)
                        op = computeOpinion(E_op, e, multi_paths)
                        E_op[e] = op
                        edges_start_at, edges_end_at, nns = update(edges_start_at, edges_end_at, nns, e)
                        Omega_X[e] = opinion_to_beta(op)
                    E_X1.pop(e, None)
                #case 4
                else:
                    # print 'CASE 4-------------------------------------------'
                    # print "start---"
                    multi_paths = []

                    for e_nn in edges_end_at[e[0]] + edges_start_at[e[1]]:
                        multi_paths.append([e_nn])
                    # print "*********** begin finding path"
                    # t_0=time.time()
                    paths = []
                    path = []
                    findPaths(nns, e)
                    # print i,time.time()-t_0
                    # print "*********** end finding path"
                    multi_paths += paths
                    if multi_paths != []:
                        paths = []
                        total_opinion = computeOpinion(E_op, e, multi_paths)
                        E_op[e] = total_opinion
                        edges_start_at, edges_end_at, nns = update(edges_start_at, edges_end_at, nns, e)
                        Omega_X[e] = opinion_to_beta(total_opinion)
                        E_X1.pop(e, None)
                    else:
                        Omega_X[e] = (alpha0,beta0)
                        # alpha_u = randint(1, T)
                        # beta_u = randint(1, T)
                        # Omega_X[e] = (alpha_u, beta_u)
                        E_op[e] = beta_to_opinion2(alpha0,beta0)
                        edges_start_at, edges_end_at, nns = update(edges_start_at, edges_end_at, nns, e)
                        E_X1.pop(e, None)
                t3=time.time()
                i+=1
    elif flag == 1:
        while (len(E_X1) != 0):
            for e in E_X1.keys():
                # print 'testing edge',e
                # source, target = e
                # print "*********** begin finding path"
                paths = []
                path = []
                findPaths(nns, e)
                # print "*********** end finding path"
                # global paths
                # print e, paths
                if paths != []:
                    total_opinion = computeOpinion(E_op, e, paths)
                    E_op[e] = total_opinion
                    edges_start_at, edges_end_at, nns = update(edges_start_at, edges_end_at, nns, e)
                    # G.add_edge(source, target, opinion = total_opinion)
                    Omega_X[e] = opinion_to_beta(total_opinion)
                    paths = []  # set global variable paths to [] to ensure next function call of findPaths()
                    # E_X1.remove(e)
                    E_X1.pop(e, None)
                else:
                    raise Exception('no path found on edge:', e)
    else:
        raise Exception('wrong flag parameter')
    # print G.number_of_nodes()
    # print G.number_of_edges()
    # print "SL is Done"
    paths = []
    path = []
    return Omega_X

"""=Concensus operator """
def fusion(o1, o2):
    o = [0, 0, 0, 0]
    o[0] = (o1[0] * o2[2] + o2[0] * o1[2]) / (o1[2] + o2[2] - o1[2] * o2[2])
    o[1] = (o1[1] * o2[2] + o2[1] * o1[2]) / (o1[2] + o2[2] - o1[2] * o2[2])
    o[2] = (o1[2] * o2[2]) / (o1[2] + o2[2] - o1[2] * o2[2])
    o[3] = o1[3]
    s = o[0] + o[1] + o[2]
    o[0] = o[0] / s
    o[1] = o[1] / s
    o[2] = o[2] / s
    return o


"""
INPUT: o1, o2 have the format (-belief, disbelief, uncertainty, base)
==Discounting operator
"""


def transitivity(o1, o2):
    o = [0, 0, 0, 0]
    o[0] = o1[0] * o2[0]
    o[1] = o1[0] * o2[1]
    o[2] = o1[1] + o1[2] + o1[0] * o2[2]
    o[3] = o2[3]

    s = o[0] + o[1] + o[2]
    o[0] = o[0] / s
    o[1] = o[1] / s
    o[2] = o[2] / s
    return o


def obs_to_omega(obs):
    '''
    get Omega based on observation dict
    input:
        obs: (dict) key is edge, value is a list of obs of the edge
    output:
        Omega: (dict) key is the edge (tuple), value is (alpha,beta)
    '''
    Omega = {}
    for e in obs:
        obs_list = obs[e]
        alpha = np.count_nonzero(obs_list) + 0.001
        beta = len(obs_list) - np.count_nonzero(obs_list) + 0.001
        if e not in Omega:
            Omega[e] = (alpha, beta)
        else:
            raise Exception("HAHA ERROR: Duplicate Edges")
    return Omega


def computeOpinion(E_op, e, multi_paths):
    '''
    input:
        G: a graph, networkx graph object
        e: is (source, target), whose opinion you want to predict
        multi_paths: [[],[]] 2D list, each sublist is a edge path from source to target
    output:
        total_opinion: [b,d,u,a]
    '''
    if multi_paths == []:
        raise error('Empty Path')
        return
    else:
        paths_opinion_list = []
        for p in multi_paths:
            if len(p) >= 2:
                opinion_path = []
                for arc in p:
                    o = E_op[arc]
                    # o = G[arc[0]][arc[1]]['opinion']
                    opinion_path.append(o)
                tran_op = opinion_path[0]
                for i in range(1, len(opinion_path)):
                    tran_op = transitivity(tran_op, opinion_path[i])
                paths_opinion_list.append(tran_op)
            else:
                arc = p[0]
                o = E_op[arc]
                # o = G[arc[0]][arc[1]]['opinion']
                paths_opinion_list.append(o)
        total_opinion = paths_opinion_list[0]
        for i in range(1, len(paths_opinion_list)):
            total_opinion = fusion(total_opinion, paths_opinion_list[i])
        return total_opinion

# if __name__ == '__main__':
#     #test()
#     test1()
#     #pipeline(V,E,Obs)
