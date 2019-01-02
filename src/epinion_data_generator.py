__author__ = 'Feng Chen'
import pickle
from math import *
import numpy as np
import copy
import random
from scipy.stats import beta
import os
from SL_inference import *


global paths
paths = [] 
global path
path = []


def findPaths(G, source, target, threshold=10):
    arcs = []
    for nei in nx.neighbors(G,source):
        arc = (source,nei)
        length = len(path)
        #path_copy = copy.copy(path)
        count = 0
        for (i,j) in path:
            if nei != i and nei != j:
                count += 1
        if count == length:    
            arcs.append(arc)
        
    for arc in arcs:
        if arc[1] == target and len(path) + 1 <= threshold:
            path_copy = copy.copy(path)
            path_copy.append(arc)
            paths.append(path_copy)
        elif arc[1] != target and len(path) + 1 <= threshold:
            path.append(arc)
            findPaths(G,arc[1],target)
            path.remove(arc)


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
def simulation_data_generator(V, E, rates = [0.05, 0.1, 0.15, 0.25, 0.3, 0.4, 0.5], T = 50):
    datasets = {}
    len_E = len(E)
    Omega = {}
    #realizations = 10
    realizations = 10
    for e in E:
        Omega[e] = generate_a_random_opinion()
    #for each rate generate 10 samples of simulation data
    for rate in rates:
        rate_datasets = []
        for real_i in range(realizations):
            len_X = int(round(len_E * rate))
            #random sample edges saved as a list 
            rate_X = [E[i] for i in np.random.permutation(len_X)[:len_X]]
            #rate_X = random.sample(E, int(round(len_X*rate)))        
            #rate_omega_X is a dict, key is edge, value is a tuple (alpha,beta)
            rate_omega_X = SL_prediction(V, E, Omega, rate_X, flag=1)
            rate_omega = copy.deepcopy(Omega)
            for e, alpha_beta in rate_omega_X.items():
                # change the rate opinion to predicted opinion
                Omega[e] = alpha_beta
            rate_Obs = {}
            for e in E:
                e_alpha, e_beta = Omega[e]
                #for each edge generate observations based on beta_dist(alpha,beta)
                #so the testing edge has ground_truth 
                rate_Obs[e] = beta.rvs(e_alpha, e_beta, 0, 1, T)
            datasets[(rate, real_i)] = [V, E, rate_omega, rate_Obs, rate_X]
    return datasets
 
 
"""
Generate simulation datasets with different graph sizes and rates
"""
def multiple_simulation_data_generator():
    graph_sizes = [500, 1000, 5000, 10000, 47676]
    rates = [0.05, 0.1, 0.15, 0.25, 0.3, 0.4, 0.5]
    T = 10
    for graph_size in graph_sizes[2:3]:
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
INPUT
V = [0, 1, ...] is a list of vertex ids
E: a list of pairs of vertex ids
Obs: a dictionary with key edge and its value a list of congestion observations of this edge from t = 1 to t = T
Omega: a dictionary with key edge and its value a pair of alpha and beta
E_X: a list of pairs of vertex ids whose opinions will be predicted.
 
OUTPUT
omega_X: {edge: (alpha, beta), ...}, where edge \in E_X and is represented by a pair of vertex ids
"""
#def SL_prediction(V, E, Omega, E_X):
    # omega_X = {e: alpha_beta for e, alpha_beta in Omega.items() if e in E_X}
#    return omega_X
     
def SL_prediction(V,E,Omega,E_X, flag = 0):
    '''
    flag == 0:  if no path between (i,j), fuse opinions of neighbors' edges of i and j 
         == 1:  not consider fusing neighbor edges
    '''
    #G = nx.Graph()
    if flag == 0:
        G = nx.Graph()   #used for traffic dataset
    elif flag == 1:
        G = nx.DiGraph()  # used for trustness dataset
    else:
        raise Exception('Wrong flag argument')

    E_known = [e for e in E if e not in E_X]
    G.add_nodes_from(V)
    Omega_X = {}
    print 'number of known edges',len(E_known)

    for e in E_known:
        source, target = e
        alpha, beta = Omega[e]
        op = beta_to_opinion(alpha,beta)
        G.add_edge(source,target,opinion=op)
    print 'number of training edges:',len(G.edges())

    Omega_X = {}

    if flag == 0:
        # used for traffic dataset, not consider direction
        print 'flag == 0'
        while(len(E_X)!=0):
            for e in E_X:
                source, target = e
                #case 1
                if source not in G.nodes() and target not in G.nodes():
                    print 'not in G',source,target
                    print len(E_X)
                    continue
                #case 2
                elif source not in G.nodes():# or target not in G.nodes():
                    G.add_node(source)
                    neis = G.neighbors(target)
                    multi_paths = []
                    neighbor_path = []
                    for nei in neis:
                        if (target,nei) in G.edges():
                            neighbor_path.append((target,nei))
                        if (nei, target)in G.edges():
                            neighbor_path.append((nei,target))
                    if neighbor_path == []:
                        raise Exception('errrror 1')
                    else:
                        multi_paths.append(neighbor_path)
                    op = computeOpinion(G,e,multi_paths)
                    G.add_edge(source,target,opinion = op)
                    Omega_X[e] = opinion_to_beta(op)
                    E_X.remove(e)
                #case 3
                elif target not in G.nodes(): 
                    G.add_node(target)
                    neis = G.neighbors(source)
                    multi_paths = []
                    neighbor_path = []
                    for nei in neis:
                        if (source,nei) in G.edges():
                            neighbor_path.append((source,nei))
                        if (nei, source)in G.edges():
                            neighbor_path.append((nei,source))
                    if neighbor_path == []:
                        raise Exception('errrror 1')
                    else:
                        multi_paths.append(neighbor_path)
                    op = computeOpinion(G,e,multi_paths)
                    G.add_edge(source,target,opinion = op)
                    Omega_X[e] = opinion_to_beta(op)
                    E_X.remove(e)
                #case 4
                else:
                    findPaths(G,source, target)
                    global paths
                    print e, paths
                    if paths != []:
                        total_opinion = computeOpinion(G,e,paths)
                        G.add_edge(source, target, opinion = total_opinion)
                        Omega_X[e] = opinion_to_beta(total_opinion)
                        paths = []
                        E_X.remove(e) 
                    else:
                        neis = G.neighbors(source) + G.neighbors(target)
                        multi_paths = []
                        neighbor_path = []
                        for nei in neis:
                            if (source,nei) in G.edges():
                                neighbor_path.append((source,nei))
                            if (nei, source)in G.edges():
                                neighbor_path.append((nei,source))
                            if (target,nei) in G.edges():
                                neighbor_path.append((target,nei))
                            if (nei,target) in G.edges():
                                neighbor_path.append((nei,target))
                        if neighbor_path == []:
                            raise Exception('errrror 1')
                        else:
                            multi_paths.append(neighbor_path)
                            total_opinion = computeOpinion(G,e,multi_paths)
                            G.add_edge(source, target, opinion = total_opinion)
                            Omega_X[e] = opinion_to_beta(total_opinion)
                            E_X.remove(e) 
    elif flag == 1:
        #used for trustness network
        print 'testing case:',len(E_X)
        nopath_count = 0
        while(len(E_X)!=0):
            for e in E_X:
                #print 'testing edge',e
                source, target = e
                findPaths(G,source, target)
                global paths
                print e, paths
                if paths == []:
                    nopath_count += 1.0
                    E_X.remove(e)
                    #raise Exception('no path found on edge:',e)
                else:
                    total_opinion = computeOpinion(g,e,paths)
                    G.add_edge(source, target, opinion = total_opinion)
                    Omega_x[e] = opinion_to_beta(total_opinion)
                    paths = []
                    E_x.remove(e)                

        print 'nopath',nopath_count
    else:
        raise Exception('wrong flag parameter')
    print G.number_of_nodes()
    print G.number_of_edges()
    return Omega_X
 


def test():
    G = nx.DiGraph()
    V = [1,2,3,4,5,6,7,8,9,10,11]
    E = [(1,2),(1,3),(2,4),(3,5),(2,6),(3,6),(6,7),(7,8),(4,9),(5,10),(7,9),(8,10),(7,11),(8,11)]
    G.add_nodes_from(V)
    G.add_edges_from(E)    
    Omega = {}
    for e in G.edges():
        alpha = randint(2,7)
        beta = 10 - alpha
        Omega[e] = [alpha,beta]
    E_X = [(4,9),(3,6),(7,9),(8,10)]
    #print Omega
    #nx.draw_networkx(G)
    Omega_X = SL_prediction(V,E,Omega,E_X,flag=1)    
    for e in Omega_X:
        print '\nedge',e
        print 'predicted opinion',Omega_X[e]
        alpha, beta = Omega[e]
        print 'true opinion',beta_to_opinion(alpha,beta)
    #print Omega_X
 

if __name__=='__main__':
    multiple_simulation_data_generator()
    #pkl_file = open("data/trust-analysis/nodes-{0}-rate-{1}-realization-{2}-data.pkl".format(graph_size, rate, real_i), 'wb')
    #pickle.dump(dataset, pkl_file)
    #pkl_file.close()

