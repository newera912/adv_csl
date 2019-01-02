__author__ = ''

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
import multiprocessing
threshold0=5
# # global paths
# paths = []
# # global path
# path = []

class FindPath():
    def __init__(self, nns, e,threshold):
        self.__path = []
        self.__paths = []
        self.nns = nns
        self.threshold=threshold
        # print self.pid,self.jobs
        # self.path.append(pid)
        self.findpath(e)

    # def run(self):
    #     self.findpath()
    #     return

    def findpath(self,e):
        # if len(paths) > 5: return
        # global paths, path
        # print source, target, threshold
        arcs = []
        source, target = e
        for nei in self.nns[source]:
            # for nei in nx.neighbors(G,source):
            arc = (source, nei)
            length = len(self.__path)
            # path_copy = copy.copy(path)
            count = 0
            for (i, j) in self.__path:
                if nei != i and nei != j:
                    count += 1
            if count == length:
                arcs.append(arc)
        for arc in arcs:
            # print path
            if arc[1] == target and len(self.__path) + 1 <= self.threshold:
                path_copy = copy.copy(self.__path)
                path_copy.append(arc)
                self.__paths.append(path_copy)
                # if len(paths)>5: return
            elif arc[1] != target and len(self.__path) + 1 <= self.threshold:
                # if len(paths) > 5: return
                self.__path.append(arc)
                self.findpath((arc[1], target))
                self.__path.remove(arc)

class Consumer(multiprocessing.Process):
    def __init__(self, task_queue, result_queue,id):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.id=id
    def run(self):
        self.name=str(self.id)
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

def nns_checked(V_checked,nns):
    for v in nns:
        if V_checked.has_key(v): return True
        else: continue
    return False


def get_neighbor_paths(v,graph_nns,V_checked,hop=3):
    neighbor_path={}
    for i in range(hop):
        if len(neighbor_path)==0:
            for v_n in graph_nns[v]:
                if V_checked.has_key(v_n):
                    neighbor_path[(v_n,)]=(i+1)
        else:
            for n_path in neighbor_path.keys():
                if neighbor_path[n_path] >0:
                    has_more_neighbor=False
                    for v_nn in graph_nns[n_path[-1]]:
                        if (v_nn not in n_path) and (V_checked.has_key(v_nn)) and v_nn!=v:
                            has_more_neighbor=True
                            neighbor_path[n_path+(v_nn,)]=(i+1)
                    if has_more_neighbor==True:
                        neighbor_path.pop(n_path, None)
                        continue

    paths=[list(k) for k,v in neighbor_path.items() if v>0]

    return paths

class Task_subSL(object):
    #                  X,V_op,V_checked, nns,alpha0,beta0,Threshold
    def __init__(self, E_X, V_op, V_checked, nns, graph_nns, alpha0, beta0, threshold):
        self.E_X1 = {e:1 for e in E_X}
        self.V_op = V_op
        self.V_checked = V_checked
        self.nns = nns
        self.graph_nns= graph_nns
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.threshold = threshold

    def __call__(self):
        # this is the place to do your work
        # time.sleep(0.1) # pretend to take some time to do our work
                      # omega, y_t, Y, X, edge_up_nns, edge_down_nns, p0, R, psl, approx, report_stat

        org_e_x=len(self.E_X1)
        Omega_X={}
        max_iter=5
        x_size=0
        for iter in range(max_iter):
            # print "#",int(multiprocessing.current_process().name),"  Iterate --- ",iter
            X = self.E_X1.keys()
            if len(X)==0:
                # print "X is zero..."
                return Omega_X,self.E_X1,self.V_op,self.V_checked,self.nns
            elif x_size==len(X):
                # print "No changes..."
                # print len(self.E_X1), "/", org_e_x
                for e in self.E_X1:
                    Omega_X[e] = (self.alpha0, self.beta0)
                return Omega_X, self.E_X1, self.V_op, self.V_checked, self.nns
            else:
                # print "Updated ..."
                x_size=len(X)
            # X = self.E_X1.keys()
            for v in X:
                # if len(E_X1)%1000==0:
                #     print len(E_X1),t0-t3,"****************************************************************************"
                #     t0=time.time()

                # case 1
                if not nns_checked(self.V_checked,self.graph_nns[v]):
                    # print "case 1",e
                    continue

                #case 2
                else:
                    self.V_checked[v] = 1
                    neighbor_path = []
                    neighbor_path = get_neighbor_paths(v,self.graph_nns,self.V_checked,hop=3)


                    # multi_paths = []
                    if neighbor_path == []:
                        Omega_X[v] = (self.alpha0,self.beta0)
                        # alpha_u=randint(1, T)
                        # beta_u=randint(1, T)
                        # Omega_X[e] = (alpha_u, beta_u)
                        # opi = beta_to_opinion(1.0,1.0)
                        self.V_op[v] = beta_to_opinion2(self.alpha0,self.beta0)
                        self.nns = update_nns(self.graph_nns, self.nns,self.V_checked, v)
                        # raise Exception('dead edge encounted.')
                    else:
                        # multi_paths.append(neighbor_path)
                        op = computeOpinion(self.V_op, v, neighbor_path)
                        self.V_op[v] = op
                        self.nns = update_nns(self.graph_nns, self.nns, self.V_checked, v)
                        # G.add_edge(source,target,opinion = op)
                        Omega_X[v] = opinion_to_beta(op)
                    # E_X1.remove(e)
                    self.E_X1.pop(v, None)


        if len(self.E_X1)!=0:
            print "?????",len(self.E_X1),"/",org_e_x
            for e in self.E_X1:
                Omega_X[e]=(self.alpha0,self.beta0)
        return Omega_X,self.E_X1,self.V_op,self.V_checked, self.nns

    def __str__(self):
        return '%s' % (self.p0)

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
        if arc[1] == target and len(path) + 1 <= threshold0:
            path_copy = copy.copy(path)
            path_copy.append(arc)
            paths.append(path_copy)
            # if len(paths)>5: return
        elif arc[1] != target and len(path) + 1 <= threshold0:
            # if len(paths) > 5: return
            path.append(arc)
            findPaths(nns, (arc[1], target))
            path.remove(arc)


def get_v_edge_nns(V, E):
    nns = {}
    g_nns={}
    for e in E:
        start_v = e[0]
        end_v = e[1]
        if g_nns.has_key(start_v):
            g_nns[start_v].append(end_v)
        else:
            g_nns[start_v] = [end_v]
    # if not V.has_key(start_v) or not V.has_key(end_v): continue
    for v in V:
        for vv in g_nns[v]:
            if V.has_key(vv):
                if nns.has_key(v): nns[v].append(vv)
                else: nns[v]=[vv]
    return nns,g_nns


def update_nns(graph_nns,nns,V_checked, v):
    # nns[v]=graph_nns[v]
    for v_nns in graph_nns[v]:
        if V_checked.has_key(v_nns):
            if nns.has_key(v) :
                nns[v].append(v_nns)
            else:
                nns[v] = [v_nns]

            if nns.has_key(v_nns):
                nns[v_nns].append(v)
            else:
                nns[v_nns] = [v]


    return nns


def SL_prediction_multiCore_node(V, E, Obs, Omega, E_X):
    # print "Path len:",threshold
    global paths, path
    T=len(Obs.values()[0])
    Threshold=3
    #when alpha=3.0, beta=3.0, then the u=1/3,d=1/3,u=1/3
    alpha0=1.0
    beta0=1.0
    # E_X1 = copy.deepcopy(E_X)
    E_X1 = {v:1 for v in E_X}
    V_op = {}
    V_checked = {}
    for v in V:
        if not E_X1.has_key(v):
            alpha, beta = Omega[v]
            V_op[v] = beta_to_opinion2(alpha, beta)
            V_checked[v] = 1

    # print V_checked.keys(), E_op.keys()

    nns,graph_nns = get_v_edge_nns(V_checked,E)
    tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()
    num_consumers = 30  # We only use 5 cores.
    # if len(E_X1)<45 : num_consumers=len(E_X1)
    print 'Creating %d consumers' % num_consumers
    consumers = [Consumer(tasks, results,i) for i in range(num_consumers)]
    for w in consumers:
        w.start()
    num_jobs = 0
    org_count=len(E_X1)
    m = int(np.ceil(float(len(E_X1)) / num_consumers))
    # print m,np.cwil(float(len(E_X1)) / num_consumers),org_count
    for (start_k, end_k)  in chunks(E_X1,m):
        # (start_k, end_k) = (int(m * i), int(m * (i + 1)))
        X=E_X1.keys()[start_k:end_k]
        org_count=org_count-len(X)
        tasks.put(Task_subSL(X,V_op,V_checked, nns,graph_nns,alpha0,beta0,Threshold))
        num_jobs+=1
    if org_count!=0: print "Errorrrrrrrrrrrrrrrrrrrrrrrrrrrrr"
    for i in range(num_consumers):
        tasks.put(None)

    """ Combine results"""
    Omega_X = {}
    left_E_X1={}
    All_E_op={}
    All_V_checked={}
    All_edges_start_at={}
    All_edges_end_at ={}
    All_nns={}


    while num_jobs:
        Omega_X_t, E_X1_t, E_op_t, V_checked_t, nns_t=results.get()
        Omega_X.update(Omega_X_t)
        num_jobs-=1.0

    return Omega_X



def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield i,i+n

def combine_dict(A,B):
    for k,v in B.items():
        if A.has_key(k):
            A[k].extend(v)
        else:
            A[k]=v
    return A



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


def computeOpinion(V_op, v, multi_paths):
    '''
    input:
        G: a graph, networkx graph object
        v: is a node, whose opinion you want to predict
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
                    o = V_op[arc]
                    # o = G[arc[0]][arc[1]]['opinion']
                    opinion_path.append(o)
                tran_op = opinion_path[0]
                for i in range(1, len(opinion_path)):
                    tran_op = transitivity(tran_op, opinion_path[i])
                paths_opinion_list.append(tran_op)
            else:
                arc = p[0]
                o = V_op[arc]
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
