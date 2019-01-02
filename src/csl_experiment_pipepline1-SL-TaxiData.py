__author__ = 'Feng Chen'
# from multi_core_trust_inference import inference_apdm_format, list_sum, list_dot_scalar, calculate_measures
from log import Log
import os,sys
import time
import pickle
import numpy as np
import json,itertools,networkx as nx
#from SL_inference import *
#from SL_prediction import findPaths, SL_prediction
from basic_funs import *
from network_funs import *
import baseline
from feng_SL_inference import *
from multi_core_csl_inference_conflicting_evidence_new import inference_apdm_format as inference_apdm_format_conflict_evidence
from csl_traffic_conflicting_evidence_pipeline1 import inference_apdm_format
from csl_traffic_conflicting_evidence_pipeline1 import inference_apdm_format_single
#from SL_inference import *
from random import shuffle
import multiprocessing
from collections import Counter



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
def calculate_measures(true_omega_x, pred_omega_x, XX, logging,No_Obs):
    # print "X", X
    X={e:1 for e in XX if not No_Obs.has_key(e)}
    W = 2.0
    bs = []
    ds = []
    for e in X:
        b1,d1,u1,a1 = beta_to_opinion(true_omega_x[e][0], true_omega_x[e][1])
        b2,d2,u2,a2 = beta_to_opinion(pred_omega_x[e][0], pred_omega_x[e][1])
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



def main():
    # experiment_proc()
    # experiment_proc_server_SL()
    experiment_proc_server_multi_timeslot_CSL()



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
class Task_evaluate(object):
    #                   V, E, Obs, Omega, E_X, logging, method, day,T, start_t,test_ratio, ref_ratio
    def __init__(self, V, E, Obs, Omega, E_X, logging, method,day,T, start_t,test_ratio, ref_ratio,No_Obs):
        self.V = V
        self.E = E
        self.Obs =Obs
        self.Omega = Omega
        self.E_X = E_X
        self.logging = logging
        self.method = method
        self.day = day
        self.T = T
        self.start_t = start_t
        self.test_ratio = test_ratio
        self.ref_ratio = ref_ratio
        self.No_Obs = No_Obs


    def __call__(self):
        running_starttime = time.time()
        pred_omega_x = SL_prediction(self.V, self.E, self.Obs, self.Omega, copy.deepcopy(self.E_X))
        alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested = calculate_measures(
            self.Omega, pred_omega_x, self.E_X, self.logging, self.No_Obs)
        running_endtime = time.time()
        running_time = running_endtime - running_starttime
        # print "accuracy: {}, prob_mse: {}".format(accuracy, prob_mse)
        return accuracy, alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, running_time,self.method,self.day,self.T,self.start_t,self.test_ratio,self.ref_ratio


    def __str__(self):
        return '%s' % (self.p0)

class Task_Time_Window(object):
    #                   V, E, Obs, Omega, E_X, logging, method, day,T, start_t,test_ratio, ref_ratio
    def __init__(self, V, E, Obs,T,logging,method,start_t,test_ratio,edge_up_nns, edge_down_nns):
        self.V = V
        self.E = E
        self.Obs =Obs
        self.start_t = start_t
        self.T=T
        self.logging=logging
        self.method=method
        self.test_ratio=test_ratio
        self.edge_up_nns=edge_up_nns
        self.edge_down_nns=edge_down_nns



    def __call__(self):
        running_starttime = time.time()
        E_X = {}
        Has_Obs = []
        No_Obs = {}
        E_X_Obs = {}
        t_Obs = {}
        for e, e_Obs in self.Obs.items():
            t_Obs[e] = e_Obs[self.start_t:self.start_t + self.T]
            if -1.0*len(t_Obs[e]) == sum(t_Obs[e]):
                E_X[e] = 1
                No_Obs[e] = 1
            else:
                Has_Obs.append(e)

            # freq_val=fred_obs(t_Obs)
            # t_Obs[e] = map(lambda x: x if x != -1 else 0.0, t_Obs[e])
        shuffle(Has_Obs)
        rands = Has_Obs[:int(np.round(self.test_ratio * len(Has_Obs)))]
        for e in rands:
            if self.edge_up_nns.has_key(e) or self.edge_down_nns.has_key(e):
                E_X[e] = 1
                E_X_Obs[e] = 1
        for e in t_Obs.keys():
            if e not in E_X and t_Obs[e].count(-1)>0:
                freq_val=fred_obs(t_Obs[e])
                t_Obs[e] = map(lambda x: x if x != -1 else freq_val, t_Obs[e])
        Omega = calc_Omega_from_Obs(t_Obs, self.E, No_Obs,E_X)
        # E_X = {e: 1 for e in E_X}

        # print V[:10]
        accuracy, alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, running_time = evaluate(self.V, self.E, t_Obs, Omega, E_X,self.logging, No_Obs, self.method)

        running_endtime = time.time()
        running_time = running_endtime - running_starttime
        # print "accuracy: {}, prob_mse: {}".format(accuracy, prob_mse)
        return accuracy, alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, running_time,self.method,self.T,self.start_t,self.test_ratio


    def __str__(self):
        return '%s' % (self.p0)

def fred_obs(obs):
    true_obs=[i for i in obs if i!=-1.0]
    # if true_obs.count(1.0)>0: return 1.0
    # else: return 0.0
    fredObs=float(Counter(true_obs).most_common(1)[0][0])

    return fredObs

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

def calc_Omega_from_Obs(Obs, E,No_Obs,E_X):
    T = len(Obs.values()[0])
    Omega = {}
    for e in E:
        if No_Obs.has_key(e):
            Omega[e]=(1.0,1.0)
        elif E_X.has_key(e):
            temp_obs=[i for i in Obs[e] if i!=-1]
            Omega[e] = (np.sum(temp_obs) * 0.5, T - np.sum(temp_obs) * 1.0 + 0.5)
        else:
            Omega[e] = (np.sum(Obs[e]) * 0.5, T - np.sum(Obs[e]) * 1.0 + 0.5)

    return Omega

def experiment_proc_server_SL():
    logging = Log()
    data_root = "/media/apdm05/data05/GasConsumption_Data_KDD2014_Zheng/proc_data/"
    methods = ["csl", "csl-3-rules", "csl-3-rules-conflict-evidence", "sl", "base1", "base2", "base3"][3:4]

    [V, E, _] = pickle.load(open("/media/apdm05/data05/GasConsumption_Data_KDD2014_Zheng/proc_data/20130912ref_proc-0.6.pkl","rb"))
    edge_up_nns, edge_down_nns = get_edge_nns(V, E)


    days = ["20130912", "20130913", "20130914", "20130915"]
    for ref_ratio in [0.9, 0.8, 0.6][:]:
        for test_ratio in [0.1,0.2,0.3,0.4]:
            for day in days[:]:
                for T in [6][:]:
                    tasks = multiprocessing.Queue()
                    results = multiprocessing.Queue()
                    num_consumers = 14  # We only use 5 cores.
                    print 'Creating %d consumers' % num_consumers
                    consumers = [Consumer(tasks, results)
                                 for i in range(num_consumers)]
                    for w in consumers:
                        w.start()
                    num_jobs = 0
                    f = data_root + day +"ref_proc-"+str(ref_ratio)+ ".pkl"
                    # out_folder = data_root + "/"
                    logging.write(str(num_jobs)+" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    for method in methods[:]:
                        logging.write("method: {}, T: {},ref_ratio: {},Test Ratio:{}".format(method, T, ref_ratio,test_ratio))
                        logging.write(f)
                        pkl_file = open(f, 'rb')
                        [V, E, Obs] = pickle.load(pkl_file)
                        pkl_file.close()
                        for start_t in range(36,144 - T + 1)[:]: #6:00am~12:00am
                            # print "start_t", start_t
                            E_X={}
                            Has_Obs=[]
                            No_Obs={}
                            E_X_Obs={}
                            t_Obs={}
                            for e, e_Obs in Obs.items():
                                t_Obs[e]=e_Obs[start_t:start_t + T]
                                if len(t_Obs[e])==t_Obs[e].count(-1):
                                    E_X[e]=1
                                    No_Obs[e] = 1
                                else:
                                    Has_Obs.append(e)
                                # t_Obs[e] = map(lambda x: x if x != -1 else 0.0, t_Obs[e])

                            shuffle(Has_Obs)
                            rands = Has_Obs[:int(np.round(test_ratio * len(Has_Obs)))]
                            for e in rands:
                                if edge_up_nns.has_key(e) or edge_down_nns.has_key(e):
                                    E_X[e] = 1
                                    E_X_Obs[e] = 1
                            for e in t_Obs.keys():
                                if e not in E_X and t_Obs[e].count(-1) > 0:
                                    freq_val = fred_obs(t_Obs[e])
                                    t_Obs[e] = map(lambda x: x if x != -1 else freq_val, t_Obs[e])
                            Omega = calc_Omega_from_Obs(t_Obs, E,No_Obs,E_X)
                            # E_X = {e: 1 for e in E_X}
                            tasks.put(Task_evaluate(V, E, Obs, Omega, E_X, logging, method, day,T, start_t,test_ratio, ref_ratio,No_Obs))
                            num_jobs += 1

                            print ">>",len(E_X),len(No_Obs)



                        # Add a poison pill for each consumer
                        for i in xrange(num_consumers):
                            tasks.put(None)
                        accuracys = []
                        prob_mses = []
                        u_mses = []
                        b_mses = []
                        d_mses = []
                        alpha_mses = []
                        beta_mses = []
                        running_times = []

                        while num_jobs:
                            accuracy, alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, running_time, method, day,T,start_t,test_ratio,ref_ratio = results.get()
                            b_mses.append(b_mse)
                            d_mses.append(d_mse)
                            alpha_mses.append(alpha_mse)
                            beta_mses.append(beta_mse)
                            accuracys.append(accuracy)
                            prob_mses.append(prob_mse)
                            u_mses.append(u_mse)
                            running_times.append(running_time)
                            num_jobs -= 1
                            # print num_jobs
                        mu_alpha_mse = np.mean(alpha_mses)
                        sigma_alpha_mse = np.std(alpha_mses)
                        mu_beta_mse = np.mean(beta_mses)
                        sigma_beta_mse = np.std(beta_mses)
                        mu_u_mse = np.mean(u_mses)
                        sigma_u_mse = np.std(u_mses)
                        mu_b_mse = np.mean(b_mses)
                        sigma_b_mse = np.std(b_mses)
                        mu_d_mse = np.mean(d_mses)
                        sigma_d_mse = np.std(d_mses)
                        mu_accuracy = np.mean(accuracys)
                        sigma_accuracy = np.std(accuracys)
                        mu_prob_mse = np.mean(prob_mses)
                        sigma_prob_mse = np.std(prob_mses)
                        running_time = np.mean(running_times)
                        result_ = {'day':day ,'test_ratio': test_ratio,
                                   'sample_size': 1, 'T': T, 'ref_ratio': ref_ratio,
                                   'test_ratio': test_ratio, 'acc': (mu_accuracy, sigma_accuracy),
                                   'prob_mse': (mu_prob_mse, sigma_prob_mse),'alpha_mse': (mu_alpha_mse, sigma_alpha_mse), 'beta_mse': (mu_beta_mse, sigma_beta_mse), 'u_mse': (mu_u_mse, sigma_u_mse), 'b_mse': (mu_b_mse, sigma_b_mse), 'd_mse': (mu_d_mse, sigma_d_mse), 'runtime': running_time}
                        print result_
                        outf = '../output/test/{}_results-server-BJTaxi-June4.json'.format(method)
                        with open(outf, 'a') as outfp:
                            outfp.write(json.dumps(result_) + '\n')
                            sys.stdout.write(str(num_jobs)+" ")
                            if num_jobs%100==0:
                                sys.stdout.write("\n")

def experiment_proc_server():
    logging = Log()
    data_root = "/media/apdm05/data05/GasConsumption_Data_KDD2014_Zheng/proc_data/"
    methods = ["csl", "csl-3-rules", "csl-3-rules-conflict-evidence", "sl", "base1", "base2", "base3"][:1]

    [V, E, _] = pickle.load(open("/media/apdm05/data05/GasConsumption_Data_KDD2014_Zheng/proc_data/20130912ref_proc-0.6.pkl","rb"))
    edge_up_nns, edge_down_nns = get_edge_nns(V, E)


    days = ["20130912", "20130913", "20130914", "20130915"]
    for day in days[:1]:
        for ref_ratio in [0.9, 0.8, 0.6][:]:
            for test_ratio in [0.1]:
                for T in [6][:]:
                    f = data_root + day +"ref_proc-"+str(ref_ratio)+ ".pkl"
                    # out_folder = data_root + "/"
                    logging.write(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    for method in methods[:]:
                        logging.write("method: {}, T: {},ref_ratio: {},Test Ratio:{}".format(method, T, ref_ratio,test_ratio))
                        logging.write(f)
                        pkl_file = open(f, 'rb')
                        [V, E, Obs] = pickle.load(pkl_file)
                        pkl_file.close()
                        accuracys = []
                        prob_mses = []
                        running_times = []
                        for start_t in range(30,144 - T + 1)[:]: #6:00am~12:00am
                            # print "start_t", start_t
                            E_X={}
                            Has_Obs=[]
                            No_Obs={}
                            E_X_Obs={}
                            t_Obs={}
                            for e, e_Obs in Obs.items():
                                t_Obs[e]=e_Obs[start_t:start_t + T]
                                if len(t_Obs[e])==t_Obs[e].count(-1):
                                    E_X[e]=1
                                    No_Obs[e] = 1
                                else:
                                    Has_Obs.append(e)
                                t_Obs[e] = map(lambda x: x if x != -1 else 0.0, t_Obs[e])
                            shuffle(Has_Obs)
                            rands = Has_Obs[:int(np.round(test_ratio * len(Has_Obs)))]
                            for e in rands:
                                if edge_up_nns.has_key(e) or edge_down_nns.has_key(e):
                                    E_X[e] = 1
                                    E_X_Obs[e] = 1
                            Omega = calc_Omega_from_Obs(t_Obs, E,No_Obs)
                            E_X = {e: 1 for e in E_X}
                            # print V[:10]
                            accuracy, alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, running_time=evaluate(V, E, t_Obs, Omega, E_X, logging,No_Obs, method)
                            accuracys.append(accuracy)
                            prob_mses.append(prob_mse)
                            running_times.append(running_time)
                            print "StartTime:{},running time:{}, MSE:{}".format(start_t,running_time,prob_mse)


                        mu_accuracy = np.mean(accuracys)
                        sigma_accuracy = np.std(accuracys)
                        mu_prob_mse = np.mean(prob_mses)
                        sigma_prob_mse = np.std(prob_mses)
                        running_time = np.mean(running_times)
                        result_ = {'day':day ,'test_ratio': test_ratio,
                                   'sample_size': 1, 'T': T, 'ref_ratio': ref_ratio,
                                   'test_ratio': test_ratio, 'acc': (mu_accuracy, sigma_accuracy),
                                   'prob_mse': (mu_prob_mse, sigma_prob_mse), 'runtime': running_time}
                        print result_
                        outf = '../output/test/{}_results-server-BJTaxi.json'.format(method)
                        with open(outf, 'a') as outfp:
                            outfp.write(json.dumps(result_) + '\n')

def experiment_proc_server_multi_timeslot_CSL():
    logging = Log()
    data_root = "/media/apdm05/data05/GasConsumption_Data_KDD2014_Zheng/proc_data/"
    data_root= "/network/rit/lab/ceashpc/adil/data/bjtaxi/"
    methods = ["csl", "csl-3-rules", "csl-3-rules-conflict-evidence", "sl", "base1", "base2", "base3"][:1]

    [V, E, _] = pickle.load(open(data_root+"20130912ref_proc-0.6.pkl","rb"))
    edge_up_nns, edge_down_nns = get_edge_nns(V, E)


    days = ["20130912", "20130913", "20130914", "20130915"]

    for ref_ratio in [0.9, 0.8, 0.6][1:]:
        for test_ratio in [0.1,0.2,0.3,0.4][:1]:
            for day in days[:]:
                for T in [6][:]:
                    f = data_root + day +"ref_proc-"+str(ref_ratio)+ ".pkl"
                    # out_folder = data_root + "/"
                    logging.write(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    for method in methods[:]:
                        logging.write("method: {}, T: {},ref_ratio: {},Test Ratio:{}".format(method, T, ref_ratio,test_ratio))
                        logging.write(f)
                        pkl_file = open(f, 'rb')
                        [V, E, Obs] = pickle.load(pkl_file)
                        pkl_file.close()


                        tasks = multiprocessing.Queue()
                        results = multiprocessing.Queue()

                        # Start consumers
                        num_consumers = 50
                        print 'Creating %d consumers' % num_consumers
                        consumers = [Consumer(tasks, results)
                                     for i in xrange(num_consumers)]
                        for w in consumers:
                            w.start()

                        num_jobs = 0


                        running_times = []
                        for start_t in range(36,144 - T + 1)[:]: #6:00am~12:00am
                            tasks.put(Task_Time_Window(V, E, Obs,T,logging,method,start_t,test_ratio,edge_up_nns, edge_down_nns))
                            num_jobs+=1
                            # print "start_t", start_t

                        for i in xrange(num_consumers):
                            tasks.put(None)

                        accuracys = []
                        prob_mses = []
                        u_mses = []
                        b_mses = []
                        d_mses = []
                        alpha_mses = []
                        beta_mses = []
                        running_times = []
                        # Start printing results
                        while num_jobs:
                            accuracy, alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, running_time, method, T, start_t, test_ratio = results.get()
                            b_mses.append(b_mse)
                            d_mses.append(d_mse)
                            alpha_mses.append(alpha_mse)
                            beta_mses.append(beta_mse)
                            accuracys.append(accuracy)
                            prob_mses.append(prob_mse)
                            u_mses.append(u_mse)
                            running_times.append(running_time)
                            num_jobs -= 1

                        mu_alpha_mse = np.mean(alpha_mses)
                        sigma_alpha_mse = np.std(alpha_mses)
                        mu_beta_mse = np.mean(beta_mses)
                        sigma_beta_mse = np.std(beta_mses)
                        mu_u_mse = np.mean(u_mses)
                        sigma_u_mse = np.std(u_mses)
                        mu_b_mse = np.mean(b_mses)
                        sigma_b_mse = np.std(b_mses)
                        mu_d_mse = np.mean(d_mses)
                        sigma_d_mse = np.std(d_mses)
                        mu_accuracy = np.mean(accuracys)
                        sigma_accuracy = np.std(accuracys)
                        mu_prob_mse = np.mean(prob_mses)
                        sigma_prob_mse = np.std(prob_mses)
                        running_time = np.mean(running_times)
                        result_ = {'day':day ,'test_ratio': test_ratio,
                                   'sample_size': 1, 'T': T, 'ref_ratio': ref_ratio,
                                   'test_ratio': test_ratio, 'acc': (mu_accuracy, sigma_accuracy),
                                   'prob_mse': (mu_prob_mse, sigma_prob_mse),'alpha_mse': (mu_alpha_mse, sigma_alpha_mse), 'beta_mse': (mu_beta_mse, sigma_beta_mse), 'u_mse': (mu_u_mse, sigma_u_mse), 'b_mse': (mu_b_mse, sigma_b_mse), 'd_mse': (mu_d_mse, sigma_d_mse), 'runtime': running_time}
                        print result_
                        outf = '../output/test/{}_results-server-BJTaxi-June3.json'.format(method)
                        with open(outf, 'a') as outfp:
                            outfp.write(json.dumps(result_) + '\n')




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
""""csl-3-rules", "csl-3-rules-conflict-evidence"
def evaluate(V, E, Obs, Omega, E_X, logging,No_Obs, method = 'csl', psl = False, approx = False, init_alpha_beta = (1, 1), report_stat = False):
    running_starttime = time.time()
    if method == 'sl':
        pred_omega_x = SL_prediction(V, E, Obs, Omega, copy.deepcopy(E_X))
        # print pred_omega_x
    elif method == 'csl':
        pred_omega_x = inference_apdm_format_single(V, E, Obs, Omega, E_X, logging)
    elif method == 'csl-3-rules':
        print method
        # b = {e: 0 for e in E}
        b={}
        # X_b = []
        t1=time.time()
        X_b = {e: 0 for e in E if not E_X.has_key(e)}
        psl = True
        print "X_b cons",time.time()-t1
        pred_omega_x, _ = inference_apdm_format_conflict_evidence(V, E, Obs, Omega, b, X_b, E_X, logging, psl)
    elif method == 'csl-3-rules-conflict-evidence':
        # b = {e: 0 for e in E}
        b={}
        # X_b = {e: 0 for e in E if e not in E_X}
        X_b = {e: 0 for e in E if e not in E_X}
        psl = False
        pred_omega_x, _ = inference_apdm_format_conflict_evidence(V, E, Obs, Omega, b, X_b, E_X, logging, psl)
    elif method == 'base1':
        pred_omega_x = baseline.base1(V, E, Obs, Omega, E_X)
    elif method == 'base2':
        pred_omega_x = baseline.base2(V, E, Obs, Omega, E_X)
    elif method == 'base3':
        pred_omega_x = baseline.base3(V, E, Obs, Omega, E_X)
    else:
        raise Exception("Method Error")
    # print "pred_omega_x", pred_omega_x
    alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested = calculate_measures(Omega, pred_omega_x, E_X, logging,No_Obs)
    running_endtime = time.time()
    running_time = running_endtime - running_starttime
    #print "accuracy: {}, prob_mse: {}".format(accuracy, prob_mse)
    return accuracy, alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, running_time



if __name__=='__main__':
    # generateData()
    main()

