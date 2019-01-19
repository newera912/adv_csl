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
threshold0=18
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

class Task_subSL(object):

    def __init__(self, E_X,E_op,V_checked,edges_start_at, edges_end_at, nns,alpha0,beta0,threshold):
        self.E_X1 = {e:1 for e in E_X}
        self.E_op = E_op
        self.V_checked = V_checked
        self.edges_start_at = edges_start_at
        self.edges_end_at = edges_end_at
        self.nns = nns
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.threshold=threshold
    def __call__(self):
        # this is the place to do your work
        # time.sleep(0.1) # pretend to take some time to do our work
                      # omega, y_t, Y, X, edge_up_nns, edge_down_nns, p0, R, psl, approx, report_stat
        org_e_x=len(self.E_X1)
        Omega_X={}
        max_iter=5
        x_size=0
        un_omega=0
        for iter in range(max_iter):
            # print "#",int(multiprocessing.current_process().name),"  Iterate --- ",iter
            X = self.E_X1.keys()
            if len(X)==0:
                # print "X is zero..."
                return Omega_X,self.E_X1,self.E_op,self.V_checked,self.edges_start_at, self.edges_end_at, self.nns
            elif x_size==len(X):
                # print "No changes..."
                # print len(self.E_X1), "/", org_e_x
                for e in self.E_X1:
                    Omega_X[e] = (self.alpha0, self.beta0)
                    print("Omega0......................",un_omega)
                    un_omega+=1
                return Omega_X, self.E_X1, self.E_op, self.V_checked, self.edges_start_at, self.edges_end_at, self.nns
            else:
                # print "Updated ..."
                x_size=len(X)
            # X = self.E_X1.keys()
            for e in X:
                # if len(E_X1)%1000==0:
                #     print len(E_X1),t0-t3,"****************************************************************************"
                #     t0=time.time()
                source, target = e
                # case 1
                if not self.V_checked.has_key(source) and not self.V_checked.has_key(target):
                    # print "case 1",e
                    continue

                #case 2
                elif not self.V_checked.has_key(source):
                    self.V_checked[source] = 1
                    neighbor_path = []
                    for e_nn in self.edges_start_at[e[1]]:
                        neighbor_path.append(e_nn)

                    multi_paths = []
                    if neighbor_path == []:
                        Omega_X[e] = (self.alpha0,self.beta0)
                        print("Omega0......................", un_omega)
                        un_omega += 1
                        # alpha_u=randint(1, T)
                        # beta_u=randint(1, T)
                        # Omega_X[e] = (alpha_u, beta_u)
                        # opi = beta_to_opinion(1.0,1.0)
                        self.E_op[e] = beta_to_opinion2(self.alpha0,self.beta0)
                        self.edges_start_at, self.edges_end_at, self.nns = update(self.edges_start_at, self.edges_end_at, self.nns, e)
                        # raise Exception('dead edge encounted.')
                    else:
                        multi_paths.append(neighbor_path)
                        op = computeOpinion(self.E_op, e, multi_paths)
                        self.E_op[e] = op
                        self.edges_start_at, self.edges_end_at, self.nns = update(self.edges_start_at, self.edges_end_at, self.nns, e)
                        # G.add_edge(source,target,opinion = op)
                        Omega_X[e] = opinion_to_beta(op)
                    # E_X1.remove(e)
                    self.E_X1.pop(e, None)
                #case 3
                elif not self.V_checked.has_key(target):
                    self.V_checked[target] = 1
                    multi_paths = []
                    neighbor_path = []
                    for e_nn in self.edges_end_at[e[0]]:
                        neighbor_path.append(e_nn)

                    if neighbor_path == []:
                        Omega_X[e] = (self.alpha0,self.beta0)
                        print("Omega0......................", un_omega)
                        un_omega += 1
                        # alpha_u = randint(1, T)
                        # beta_u = randint(1, T)
                        # Omega_X[e] = (alpha_u, beta_u)
                        self.E_op[e] = beta_to_opinion2(self.alpha0,self.beta0)
                        self.edges_start_at, self.edges_end_at, self.nns = update(self.edges_start_at, self.edges_end_at, self.nns, e)
                    else:
                        multi_paths.append(neighbor_path)
                        op = computeOpinion(self.E_op, e, multi_paths)
                        self.E_op[e] = op
                        self.edges_start_at, self.edges_end_at, self.nns = update(self.edges_start_at, self.edges_end_at, self.nns, e)
                        Omega_X[e] = opinion_to_beta(op)
                    self.E_X1.pop(e, None)
                #case 4
                else:
                    # print 'CASE 4-------------------------------------------'
                    # print "start---"
                    multi_paths = []

                    for e_nn in self.edges_end_at[e[0]] + self.edges_start_at[e[1]]:
                        multi_paths.append([e_nn])
                    # print "*********** begin finding path"
                    # t_0=time.time()
                    # paths = []
                    # path = []
                    SL_Paths=FindPath(self.nns, e,self.threshold)
                    paths=SL_Paths._FindPath__paths
                    # print i,time.time()-t_0
                    # print "*********** end finding path",paths
                    multi_paths += paths
                    if multi_paths != []:
                        paths = []
                        total_opinion = computeOpinion(self.E_op, e, multi_paths)
                        self.E_op[e] = total_opinion
                        self.edges_start_at, self.edges_end_at, self.nns = update(self.edges_start_at, self.edges_end_at, self.nns, e)
                        Omega_X[e] = opinion_to_beta(total_opinion)
                        """E_X1.pop(e, None)"""
                    else:
                        Omega_X[e] = (self.alpha0,self.beta0)
                        # alpha_u = randint(1, T)
                        # beta_u = randint(1, T)
                        # Omega_X[e] = (alpha_u, beta_u)
                        self.E_op[e] = beta_to_opinion2(self.alpha0,self.beta0)
                        self.edges_start_at, self.edges_end_at, self.nns = update(self.edges_start_at, self.edges_end_at, self.nns, e)
                    self.E_X1.pop(e, None)

        if len(self.E_X1)!=0:
            print len(self.E_X1),"/",org_e_x
            for e in self.E_X1:
                Omega_X[e]=(self.alpha0,self.beta0)
        return Omega_X,self.E_X1,self.E_op,self.V_checked,self.edges_start_at, self.edges_end_at, self.nns

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


def SL_prediction_multiCore(V, E, Obs, Omega, E_X):
    # print "Path len:",threshold
    global paths, path
    T=len(Obs[E[0]])
    Threshold=10
    #when alpha=3.0, beta=3.0, then the u=1/3,d=1/3,u=1/3
    alpha0=3.0
    beta0=3.0
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
    print len(V_checked.keys()), len(E_op.keys())

    edges_start_at, edges_end_at, nns = get_v_edge_nns(V_checked.keys(), E_op.keys())
    tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()
    num_consumers = 10  # We only use 5 cores.
    # if len(E_X1)<45 : num_consumers=len(E_X1)
    print 'Creating %d consumers' % num_consumers
    consumers = [Consumer(tasks, results,i)
                 for i in range(num_consumers)]
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
        tasks.put(Task_subSL(X,E_op,V_checked,edges_start_at, edges_end_at, nns,alpha0,beta0,Threshold))
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
        Omega_X_t, E_X1_t, E_op_t, V_checked_t, edges_start_at_t, edges_end_at_t, nns_t=results.get()
        Omega_X.update(Omega_X_t)
        num_jobs-=1.0
        # left_E_X1.update(E_X1_t)
        # All_E_op.update(E_op_t)
        # All_V_checked.update(V_checked_t)
        # All_edges_start_at = combine_dict(All_edges_start_at,edges_start_at_t)
        # All_edges_end_at = combine_dict(All_edges_end_at,edges_end_at_t)
        # All_nns = combine_dict(All_nns,nns_t)
    # print "\n\nSL is done.....\n\n"

    # for i in range(5):
    #     for e in X:
    #         # if len(E_X1)%1000==0:
    #         #     print len(E_X1),t0-t3,"****************************************************************************"
    #         #     t0=time.time()
    #         source, target = e
    #         # case 1
    #         if not V_checked.has_key(source) and not V_checked.has_key(target):
    #             # print e
    #             continue
    #
    #         #case 2
    #         elif not V_checked.has_key(source):
    #             V_checked[source] = 1
    #             neighbor_path = []
    #             for e_nn in edges_start_at[e[1]]:
    #                 neighbor_path.append(e_nn)
    #
    #             multi_paths = []
    #             if neighbor_path == []:
    #                 Omega_X[e] = (alpha0,beta0)
    #                 # alpha_u=randint(1, T)
    #                 # beta_u=randint(1, T)
    #                 # Omega_X[e] = (alpha_u, beta_u)
    #                 # opi = beta_to_opinion(1.0,1.0)
    #                 E_op[e] = beta_to_opinion2(alpha0,beta0)
    #                 edges_start_at, edges_end_at, nns = update(edges_start_at, edges_end_at, nns, e)
    #                 # raise Exception('dead edge encounted.')
    #             else:
    #                 multi_paths.append(neighbor_path)
    #                 op = computeOpinion(E_op, e, multi_paths)
    #                 E_op[e] = op
    #                 edges_start_at, edges_end_at, nns = update(edges_start_at, edges_end_at, nns, e)
    #                 # G.add_edge(source,target,opinion = op)
    #                 Omega_X[e] = opinion_to_beta(op)
    #             # E_X1.remove(e)
    #             E_X1.pop(e, None)
    #         #case 3
    #         elif not V_checked.has_key(target):
    #             V_checked[target] = 1
    #             multi_paths = []
    #             neighbor_path = []
    #             for e_nn in edges_end_at[e[0]]:
    #                 neighbor_path.append(e_nn)
    #
    #             if neighbor_path == []:
    #                 Omega_X[e] = (alpha0,beta0)
    #                 # alpha_u = randint(1, T)
    #                 # beta_u = randint(1, T)
    #                 # Omega_X[e] = (alpha_u, beta_u)
    #                 E_op[e] = beta_to_opinion2(alpha0,beta0)
    #                 edges_start_at, edges_end_at, nns = update(edges_start_at, edges_end_at, nns, e)
    #             else:
    #                 multi_paths.append(neighbor_path)
    #                 op = computeOpinion(E_op, e, multi_paths)
    #                 E_op[e] = op
    #                 edges_start_at, edges_end_at, nns = update(edges_start_at, edges_end_at, nns, e)
    #                 Omega_X[e] = opinion_to_beta(op)
    #             E_X1.pop(e, None)
    #         #case 4
    #         else:
    #             # print 'CASE 4-------------------------------------------'
    #             # print "start---"
    #             multi_paths = []
    #
    #             for e_nn in edges_end_at[e[0]] + edges_start_at[e[1]]:
    #                 multi_paths.append([e_nn])
    #             # print "*********** begin finding path"
    #             # t_0=time.time()
    #             paths = []
    #             path = []
    #             findPaths(nns, e)
    #             # print i,time.time()-t_0
    #             # print "*********** end finding path"
    #             multi_paths += paths
    #             if multi_paths != []:
    #                 paths = []
    #                 total_opinion = computeOpinion(E_op, e, multi_paths)
    #                 E_op[e] = total_opinion
    #                 edges_start_at, edges_end_at, nns = update(edges_start_at, edges_end_at, nns, e)
    #                 Omega_X[e] = opinion_to_beta(total_opinion)
    #                 E_X1.pop(e, None)
    #             else:
    #                 Omega_X[e] = (alpha0,beta0)
    #                 # alpha_u = randint(1, T)
    #                 # beta_u = randint(1, T)
    #                 # Omega_X[e] = (alpha_u, beta_u)
    #                 E_op[e] = beta_to_opinion2(alpha0,beta0)
    #                 edges_start_at, edges_end_at, nns = update(edges_start_at, edges_end_at, nns, e)
    #                 E_X1.pop(e, None)
    #         t3=time.time()





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
