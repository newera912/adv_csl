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
from datetime import datetime
# from DataWrapper import *


# global paths
paths = []
# global path
path = []


def findPaths(nns, e, threshold=5):
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
        elif arc[1] != target and len(path) + 1 <= threshold:
            path.append(arc)
            findPaths(nns, (arc[1], target), threshold)
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
    global paths, path
    # E_X1 = copy.deepcopy(E_X)
    t0=time.time()
    E_X1 = {e:1 for e in E_X}
    E_op = {}
    V_checked = {}
    for e in E:
        if not E_X1.has_key(e):
            alpha, beta = Omega[e]
            E_op[e] = beta_to_opinion(alpha, beta)
            V_checked[e[0]] = 1
            V_checked[e[1]] = 1
    # print V_checked.keys(), E_op.keys()
    edges_start_at, edges_end_at, nns = get_v_edge_nns(V_checked.keys(), E_op.keys())
    print "E_X len:",len(E_X1)
    t1=time.time()
    print "Pre-process time",t1-t0
    Omega_X = {}
    if flag == 0:
        print 'flag == 0'
        while (len(E_X1) != 0):
            # X = [e for e in E_X1]
            t2=time.time()
            X = E_X1.keys()
            for e in X:
                print 'testing edge',e,len(E_X1),t2-t1
                source, target = e
                # case 1
                if not V_checked.has_key(source) and not V_checked.has_key(target):
                    continue
                # if source not in G.nodes() and target not in G.nodes():
                #     # print 'CASE 1-------------------------------------------'
                #     #print 'not in G',source,target
                #     #print len(E_X)
                #     # raise Exception('incomplete graph. ')
                #     continue
                # case 2
                elif not V_checked.has_key(source):
                    # elif source not in G.nodes():# or target not in G.nodes():
                    print 'CASE 2-------------------------------------------'
                    # print 'no source', but the target exists in the current graph
                    # G.add_node(source)
                    t3=time.time()
                    V_checked[source] = 1
                    neighbor_path = []
                    for e_nn in edges_start_at[e[1]]:
                        neighbor_path.append(e_nn)
                    # neis = G.neighbors(target)
                    # for nei in neis:
                    #     if (target,nei) in G.edges():
                    #         neighbor_path.append((target,nei))
                    multi_paths = []
                    if neighbor_path == []:
                        Omega_X[e] = (1.0, 1.0)
                        # opi = beta_to_opinion(1.0,1.0)
                        E_op[e] = beta_to_opinion(1.0, 1.0)
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
                    t4=time.time()
                    print "Case 2:",t4-t3
                elif not V_checked.has_key(target):
                    # elif target not in G.nodes():
                    print 'CASE 3-------------------------------------------'
                    # print 'no target', but the source exists in the current graph
                    # G.add_node(target)
                    t5=time.time()
                    V_checked[target] = 1
                    multi_paths = []
                    neighbor_path = []
                    for e_nn in edges_end_at[e[0]]:
                        neighbor_path.append(e_nn)
                    # for node in G.nodes():
                    #     if (node, source)in G.edges():
                    #         neighbor_path.append((nei,source))
                    if neighbor_path == []:
                        # raise Exception('errrror 1')
                        Omega_X[e] = (1.0, 1.0)
                        # opi = beta_to_opinion(1.0,1.0)
                        E_op[e] = beta_to_opinion(1.0, 1.0)
                        edges_start_at, edges_end_at, nns = update(edges_start_at, edges_end_at, nns, e)
                    else:
                        multi_paths.append(neighbor_path)
                        op = computeOpinion(E_op, e, multi_paths)
                        E_op[e] = op
                        edges_start_at, edges_end_at, nns = update(edges_start_at, edges_end_at, nns, e)
                        # G.add_edge(source,target,opinion = op)
                        Omega_X[e] = opinion_to_beta(op)
                    # E_X1.remove(e)
                    E_X1.pop(e, None)
                    t6=time.time()
                    print "Case 3:",t6-t5
                else:
                    print 'CASE 4-------------------------------------------'
                    # print "start---"
                    t7=time.time()
                    multi_paths = []
                    # print edges_end_at[e[0]], edges_start_at[e[1]]
                    # print "A", edges_end_at[e[0]]
                    # print "B", edges_start_at[e[1]]
                    for e_nn in edges_end_at[e[0]] + edges_start_at[e[1]]:
                        multi_paths.append([e_nn])
                    # for node in G.nodes():
                    #     if (target,node) in G.edges():
                    #         multi_paths.append([(target,node)])
                    #     if (node,source) in G.edges():
                    #         multi_paths.append([(node,source)])
                    # print "end----"
                    # print "*********** begin finding path"

                    paths = []
                    path = []
                    t10=time.time()
                    print "*********** finding path",datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    findPaths(nns, e)
                    print "finding path:",time.time()-t10

                    multi_paths += paths
                    if multi_paths != []:
                        paths = []
                        total_opinion = computeOpinion(E_op, e, multi_paths)
                        E_op[e] = total_opinion
                        edges_start_at, edges_end_at, nns = update(edges_start_at, edges_end_at, nns, e)
                        # G.add_edge(source, target, opinion = total_opinion)
                        Omega_X[e] = opinion_to_beta(total_opinion)
                        # print e
                        # E_X1.remove(e)
                        E_X1.pop(e, None)
                    else:
                        Omega_X[e] = (1.0, 1.0)
                        # opi = beta_to_opinion(1.0,1.0)
                        E_op[e] = beta_to_opinion(1.0, 1.0)
                        edges_start_at, edges_end_at, nns = update(edges_start_at, edges_end_at, nns, e)
                        # G.add_edge(source,target,opinion=opi)
                        # print e
                        # E_X1.remove(e)
                        E_X1.pop(e, None)
                        t8=time.time
                        print "Case 4:",time.time()-t8
        # t1=time.time()

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
    paths = []
    path = []
    return Omega_X


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
