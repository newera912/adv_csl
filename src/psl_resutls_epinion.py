import json
import pickle
import numpy as np
import re,os,time
import multiprocessing
from network_funs import *

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
class Task_inference(object):
    def __init__(self, graph_size, T, ratio, percent, swap_ratio, ratio_conflict, real_i,running_time_dict,result_folder):
        self.graph_size = graph_size
        self.T = T
        self.ratio = ratio
        self.percent = percent
        self.swap_ratio = swap_ratio
        self.ratio_conflict = ratio_conflict
        self.real_i = real_i
        self.running_time_dict=running_time_dict
        self.result_folder= result_folder
    def __call__(self):
        # this is the place to do your work
        # time.sleep(0.1) # pretend to take some time to do our work

        # result_file = str(self.graph_size) + '_' + str(self.ratio) + '_' + str(self.swap_ratio) + '_' + str(
        #     self.T) + '_' + str(self.percent) + '_' + str(self.ratio_conflict) + '_' + str(
        #     self.real_i) + '.txt'

        # if result_file not in exfiles:
        #     continue
        f = "/network/rit/lab/ceashpc/adil/data/csl-data/May23/{}/nodes-{}-T-{}-rate-{}-testratio-{}-swaprate-{}-confictratio-{}-realization-{}-data-X.pkl".format(
            self.graph_size, self.graph_size, self.T, self.ratio, self.percent, self.swap_ratio, self.ratio_conflict, self.real_i)


        pkl_file = open(f, 'rb')
        [_, E, Obs, E_X, _] = pickle.load(pkl_file)  # V, E, Obs, E_X, X_b
        pkl_file.close()
        E_X = {e: 1 for e in E_X}
        accuracys = []
        prob_mses = []
        u_mses = []
        b_mses = []
        d_mses = []
        alpha_mses = []
        beta_mses = []
        running_times = []
        for window in range(self.T):

            result_file = str(self.graph_size) + '_' + str(self.ratio) + '_' + str(
                self.swap_ratio) + '_' + str(self.T) + '_' + str(window) + '_' + str(self.percent) + '_' + str(self.ratio_conflict) + '_' + str(self.real_i) + '.txt'

            key = str(self.graph_size) + '_' + str(self.ratio) + '_' + str(self.swap_ratio) + '_' + str(window) + '_' + str(self.percent) + '_' + str(self.ratio_conflict) + '_' + str(self.real_i)
            t_Obs = {e: e_Obs[window:window+1] for e, e_Obs in Obs.items()}
            Omega = calc_Omega_from_Obs2(t_Obs, E)
            alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested = result_analysis(
                E_X, self.result_folder, result_file, Omega)

            if self.running_time_dict.has_key(key):
                running_time = self.running_time_dict[key]
            else:
                running_time = 0.0
            b_mses.append(b_mse)
            d_mses.append(d_mse)
            alpha_mses.append(alpha_mse)
            beta_mses.append(beta_mse)
            accuracys.append(accuracy)
            prob_mses.append(prob_mse)
            u_mses.append(u_mse)
            running_times.append(running_time)
        mu_alpha_mse = np.mean(alpha_mses)
        sigma_alpha_mse = np.std(alpha_mses)
        mu_beta_mse = np.mean(beta_mses)
        sigma_beta_mse = np.std(beta_mses)
        mu_u_mse = np.mean(u_mses)
        sigma_u_mse = np.std(u_mses)
        mu_b_mse = np.mean(b_mses)
        sigma_b_mse = np.mean(b_mses)
        mu_d_mse = np.mean(d_mses)
        sigma_d_mse = np.mean(d_mses)
        mu_accuracy = np.mean(accuracys)
        sigma_accuracy = np.std(accuracys)
        mu_prob_mse = np.mean(prob_mses)
        sigma_prob_mse = np.std(prob_mses)
        running_time = np.mean(running_times)
        # alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested = result_analysis(E_X_dict, sw_Omega, weekday, hour, refspeed, window, percent)
        # save_results(dataset,weekday,hour,refspeed,window,percent,alpha_mse,beta_mse,prob_mse,u_mse,b_mse,d_mse,prob_relative_mse,u_relative_mse,accuracy,recall_congested,recall_uncongested,running_time)

        result_ = {'network_size': self.graph_size, 'positive_ratio': self.ratio, "realization": self.real_i,
                   'sample_size': 1, 'T': self.T, 'ratio_conflict': self.ratio_conflict,
                   'test_ratio': self.percent, 'acc': (mu_accuracy, sigma_accuracy),
                   'alpha_mse': (mu_alpha_mse, sigma_alpha_mse), 'beta_mse': (mu_beta_mse, sigma_beta_mse),
                   'u_mse': (mu_u_mse,sigma_u_mse), 'b_mse': (mu_b_mse, sigma_b_mse), 'd_mse': (mu_d_mse, sigma_d_mse),
                   'prob_mse': (mu_prob_mse, sigma_prob_mse), 'runtime': running_time}
        output_file = open('../output/test/psl_results-server-June5-'+str(self.graph_size)+'-avg.json', 'a')
        output_file.write(json.dumps(result_) + '\n')
        output_file.close()
        return

    def __str__(self):
        return '%s' % (self.p0)


class Task_inference2(object):
    def __init__(self, graph_size, T, ratio, percent, swap_ratio, gamma, real_i,running_time_dict,result_folder,adv_type):
        self.graph_size = graph_size
        self.T = T
        self.ratio = ratio
        self.percent = percent
        self.swap_ratio = swap_ratio
        self.gamma = gamma
        self.real_i = real_i
        self.running_time_dict=running_time_dict
        self.result_folder= result_folder
        self.adv_type=adv_type
    def __call__(self):
        # this is the place to do your work
        # time.sleep(0.1) # pretend to take some time to do our work

        # result_file = str(self.graph_size) + '_' + str(self.ratio) + '_' + str(self.swap_ratio) + '_' + str(
        #     self.T) + '_' + str(self.percent) + '_' + str(self.ratio_conflict) + '_' + str(
        #     self.real_i) + '.txt'

        # if result_file not in exfiles:
        #     continue
        data_root = "/network/rit/lab/ceashpc/adil/data/adv_csl/Jan2/"
        f = data_root + "/{}/{}/nodes-{}-T-{}-rate-{}-testratio-{}-swaprate-{}-gamma-{}-realization-{}-data-X.pkl".format(
            self.adv_type, self.graph_size, self.graph_size, self.T, self.ratio, self.percent, self.swap_ratio,
            self.gamma, self.real_i)

        pkl_file = open(f, 'rb')
        [_, E, Obs, E_X, _] = pickle.load(pkl_file)  # V, E, Obs, E_X, X_b
        pkl_file.close()
        E_X = {e: 1 for e in E_X}
        probs = {e: [] for e in E_X}

        running_times = []
        Omega = calc_Omega_from_Obs2(Obs, E)
        for window in range(self.T):

            result_file = str(self.graph_size) + '_' + str(self.ratio) + '_' + str(
                self.swap_ratio) + '_' + str(self.T) + '_' + str(window) + '_' + str(self.percent) + '_' + str(
                self.gamma) + '_' + str(self.real_i) + '.txt'

            key = str(self.graph_size) + '_' + str(self.ratio) + '_' + str(self.swap_ratio) + '_' + str(
                self.T) + '_' + str(window) + '_' + str(self.percent) + '_' + str(self.gamma) + '_' + str(self.real_i)
            # t_Obs = {e: e_Obs[window:window+1] for e, e_Obs in Obs.items()}
            try:
                open(self.result_folder + result_file, 'r')
            except:
                continue
            probs_t = result_analysis2(E_X, self.result_folder, result_file)
            for e,prob in probs_t.items():
                if probs.has_key(e):
                    probs[e].append(prob)
            if self.running_time_dict.has_key(key):
                running_times.append(self.running_time_dict[key])
            else:
                print("no...",key)
                running_times.append(0.0)
        if len(running_times)==0:
            return
        pred_Omega=estimate_omega_x2(probs, E_X)
        alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested = calculate_measures(Omega, pred_Omega, E_X)
        running_time=np.sum(running_times)
        std_running_time = np.std(running_times)
        # alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested = result_analysis(E_X_dict, sw_Omega, weekday, hour, refspeed, window, percent)
        # save_results(dataset,weekday,hour,refspeed,window,percent,alpha_mse,beta_mse,prob_mse,u_mse,b_mse,d_mse,prob_relative_mse,u_relative_mse,accuracy,recall_congested,recall_uncongested,running_time)

        result_ = {'network_size': self.graph_size, 'adv_type': self.adv_type, 'positive_ratio': self.ratio,
                   "realization": self.real_i,
                   'sample_size': 1, 'T': self.T, 'gamma': self.gamma,
                   'test_ratio': self.percent, 'acc': (accuracy, 0.0),
                   'alpha_mse': (alpha_mse, alpha_mse), 'beta_mse': (beta_mse, beta_mse),
                   'u_mse': (u_mse, u_mse), 'b_mse': (b_mse, b_mse), 'd_mse': (d_mse, d_mse),
                   'prob_mse': (prob_mse, prob_mse), 'runtime': (running_time, std_running_time)}
        output_file = open('../output/epinions/PSL_results-server-'+str(self.graph_size)+'-Jan5-'+self.adv_type+'.json', 'a')
        output_file.write(json.dumps(result_) + '\n')
        output_file.close()
        return


    def __str__(self):
        return '%s' % (self.p0)

class Task_inference3(object):
    def __init__(self, graph_size, T, ratio, percent, swap_ratio, gamma, real_i,running_time_dict,result_folder,adv_type):
        self.graph_size = graph_size
        self.T = T
        self.ratio = ratio
        self.percent = percent
        self.swap_ratio = swap_ratio
        self.gamma = gamma
        self.real_i = real_i
        self.running_time_dict=running_time_dict
        self.result_folder= result_folder
        self.adv_type=adv_type
    def __call__(self):
        # this is the place to do your work
        # time.sleep(0.1) # pretend to take some time to do our work

        # result_file = str(self.graph_size) + '_' + str(self.ratio) + '_' + str(self.swap_ratio) + '_' + str(
        #     self.T) + '_' + str(self.percent) + '_' + str(self.ratio_conflict) + '_' + str(
        #     self.real_i) + '.txt'

        # if result_file not in exfiles:
        #     continue
        data_root = "/network/rit/lab/ceashpc/adil/data/adv_csl/Jan2/"
        f = data_root+"/{}/{}/nodes-{}-T-{}-rate-{}-testratio-{}-swaprate-{}-gamma-{}-realization-{}-data-X.pkl".format(
            self.adv_type,self.graph_size, self.graph_size, self.T, self.ratio, self.percent, self.swap_ratio, self.gamma, self.real_i)

        pkl_file = open(f, 'rb')
        [_, E, Obs, E_X, _] = pickle.load(pkl_file)  # V, E, Obs, E_X, X_b
        pkl_file.close()
        E_X = {e: 1 for e in E_X}
        probs = {e: [] for e in E_X}

        running_times = []
        Omega = calc_Omega_from_Obs2(Obs, E)
        True_Opinions=calc_Omega_from_Obs3(Obs,E_X)
        for window in range(self.T):

            result_file = str(self.graph_size) + '_' + str(self.ratio) + '_' + str(
                self.swap_ratio) + '_' + str(self.T) + '_' + str(window) + '_' + str(self.percent) + '_' + str(self.gamma) + '_' + str(self.real_i) + '.txt'

            key = str(self.graph_size) + '_' + str(self.ratio) + '_' + str(self.swap_ratio)+ '_' +str(self.T)+ '_' + str(window) + '_' + str(self.percent) + '_' + str(self.gamma) + '_' + str(self.real_i)
            # t_Obs = {e: e_Obs[window:window+1] for e, e_Obs in Obs.items()}
            try:
                open(self.result_folder + result_file, 'r')
            except:
                continue
            probs_t = result_analysis2(E_X, self.result_folder, result_file)
            for e,prob in probs_t.items():
                if probs.has_key(e):
                    probs[e].append(prob)
            if self.running_time_dict.has_key(key):
                running_times.append(self.running_time_dict[key])
            else:
                print("no...",key)
                running_times.append(0.0)
        if len(running_times)==0:
            return
        pred_Omega=estimate_omega_x2(probs, E_X)
        pred_opinions=Omega_2_opinion(pred_Omega,E_X)

        alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested = calculate_measures(Omega, pred_Omega, E_X)
        running_time=np.sum(running_times)
        std_running_time = np.std(running_times)
        # alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested = result_analysis(E_X_dict, sw_Omega, weekday, hour, refspeed, window, percent)
        # save_results(dataset,weekday,hour,refspeed,window,percent,alpha_mse,beta_mse,prob_mse,u_mse,b_mse,d_mse,prob_relative_mse,u_relative_mse,accuracy,recall_congested,recall_uncongested,running_time)

        result_ = {'network_size': self.graph_size,'adv_type':self.adv_type, 'positive_ratio': self.ratio, "realization": self.real_i,
                   'sample_size': 1, 'T': self.T, 'gamma': self.gamma,
                   'test_ratio': self.percent, 'acc': (accuracy, 0.0),
                   'alpha_mse': (alpha_mse, alpha_mse), 'beta_mse': (beta_mse, beta_mse),
                   'u_mse': (u_mse,u_mse), 'b_mse': (b_mse, b_mse), 'd_mse': (d_mse, d_mse),
                   'prob_mse': (prob_mse, prob_mse), 'runtime': (running_time,std_running_time)}
        output_file = open('../output/test/psl_results-server-'+str(self.graph_size)+'-nov12-0.2-tr-t246.json', 'a')
        output_file.write(json.dumps(result_) + '\n')
        output_file.close()
        return True_Opinions,pred_opinions,self.percent,self.ratio_conflict,self.T


    def __str__(self):
        return '%s' % (self.p0)

def read_raw_data(weekday, hour, refspeed):
    f = open('raw_data/raw_network_philly_weekday_' + str(weekday) + '_hour_' + str(hour) + '_refspeed_' + str(
        refspeed) + '.pkl', 'rb')
    [V, E, Obs] = pickle.load(f)
    f.close()
    return Obs


# def sliding_window_extract(Obs, start_t, T):
#     sw_Obs = {}
#     sw_Omega = {}
#     for e, obs in Obs.items():
#         sw_Obs[e] = [Obs[e][t] for t in range(start_t - T, start_t)]
#         n = np.sum(sw_Obs[e])
#         sw_Omega[e] = (n + 1, T - n + 1)
#     return sw_Omega, sw_Obs
def sliding_window_extract(Obs, start_t, window_size = 1):
    sw_Obs = {}
    sw_Omega = {}
    for e, obs in Obs.items():
        sw_Obs[e] = [Obs[e][t] for t in range(start_t-window_size, start_t)]
        n = np.sum(sw_Obs[e])
        sw_Omega[e] = (n+1,window_size-n+1)
    return sw_Omega, sw_Obs

def read_E_X_dict(weekday):
    E_X_dict = {}
    E_X_f = open('philly_traffic_day' + str(weekday) + '/easy/groovy/results/E_X.json', 'r')
    for line in E_X_f:
        [(key, value)] = json.loads(line).items()
        if key not in E_X_dict:
            E_X_dict[key] = value
    return E_X_dict


def read_running_time(file):
    running_time_dict = {}
    for weekday in range(1):
        f = open(file, 'r')
        for line in f:
            try:
                result = json.loads(line)
                # print result.keys()[0]
            except:
                print "time",line
            key = result.keys()[0]
            running_time = result[key]
            if key not in running_time_dict:
                running_time_dict[key] = running_time
    return running_time_dict

def estimate_omega_x(ps, X):
    omega_x = {}
    strategy = 1 # 1: means we consider p values as binary observations and use them to estimate alpha and beta.
    if strategy == 1:
        for e in X:
            data = [round(p_t) for p_t in [ps[e]]]
            # data = [(p_t[e]) for p_t in ps]
            alpha1 = np.sum(data) +1.0
            beta1 = len(data) - np.sum(data) +1.0
            omega_x[e] = (alpha1, beta1)
    return omega_x

def estimate_omega_x2(ps, X):
    omega_x = {}
    strategy = 1 # 1: means we consider p values as binary observations and use them to estimate alpha and beta.
    if strategy == 1:
        for e in X:
            data = [prob_2_binary2(p_t) for p_t in ps[e]]
            # data = [(p_t[e]) for p_t in ps]
            alpha1 = np.sum(data)
            beta1 = len(data) - np.sum(data)
            omega_x[e] = (alpha1, beta1)
    return omega_x

def prob_2_binary2(val):
    return val
    # if val >= 0.2:
    #     return 1
    # else:
    #     return 0

def result_analysis(E_X, result_folder,result_file, sw_Omega ):

    f = open(result_folder+result_file, 'r')
    lines = f.readlines()
    Omega_X = {}
    pred_belief = {}
    for line in lines[1:-1]:
        fields = re.split('\'|\[|\]', line)
        #edge = fields[1].split('_')
        try :
            source = int(fields[1])
            target = int(fields[3])
        except:
            print line,fields
            time.sleep(1000)
        e = (source, target)
        pred = float(fields[5])
        # print e, pred
        if not E_X.has_key(e):
            continue
        else:
            alpha = pred
            '''
            if e not in Omega_X:
                if alpha == 0:
                    Omega_X[e] = (1,2)
                elif alpha == 1:
                    Omega_X[e] = (2,1)
                else:
                    Omega_X[e] = (alpha,1.0-alpha)
            '''
            if e not in pred_belief:
                pred_belief[e] = alpha

    count = 0.0
    for e in E_X:
        if not pred_belief.has_key(e):
            count += 1
            # Omega_X[tuple(e)] = (1.0,1.0)
            pred_belief[tuple(e)] = 0.5
    # print 'Not Predicted', count
    pred_belief=estimate_omega_x(pred_belief,E_X)
    # alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested = calculate_measures(sw_Omega,Omega_X,E_X)
    # return alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested
    alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested = calculate_measures(sw_Omega, pred_belief, E_X)
    return alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested


def result_analysis2(E_X, result_folder,result_file):


    f = open(result_folder+result_file, 'r')

    lines = f.readlines()
    Omega_X = {}
    pred_belief = {}
    for line in lines[1:-1]:
        fields = re.split('\'|\[|\]', line)
        #edge = fields[1].split('_')
        try :
            source = int(fields[1])
            target = int(fields[3])
        except:
            print ">>>",line,fields

        e = (source, target)
        pred = float(fields[5])
        # print e, pred
        if not E_X.has_key(e):
            continue
        else:
            alpha = pred
            '''
            if e not in Omega_X:
                if alpha == 0:
                    Omega_X[e] = (1,2)
                elif alpha == 1:
                    Omega_X[e] = (2,1)
                else:
                    Omega_X[e] = (alpha,1.0-alpha)
            '''
            if e not in pred_belief:
                pred_belief[e] = alpha

    count = 0.0
    for e in E_X:
        if not pred_belief.has_key(e):
            count += 1
            # Omega_X[tuple(e)] = (1.0,1.0)
            pred_belief[tuple(e)] = 0.5
    # pred_belief=estimate_omega_x(pred_belief,E_X)
    # alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested = calculate_measures(sw_Omega,Omega_X,E_X)
    # return alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested

    return pred_belief


def trim(val):
    if val < 0:
        return 0
    elif val > 1:
        return 1
    else:
        return val


def beta_to_opinion(alpha, beta, W=2.0, a=0.5):
    '''
    compute opinion based on hyperparameters of beta distribution
    '''
    b = trim((alpha - W * a) / float(alpha + beta))
    d = trim((beta - W * (1 - a)) / float(alpha + beta))
    u = trim((W) / float(alpha + beta + W))
    return [b, d, u, a]


def Omega_2_opinion(pred_omega,E_X):
    W=2.0
    a=0.5
    # T = len(Obs.values()[0])
    Omega = {}
    Opinions={}
    for e in E_X.keys():
        r=np.abs(pred_omega[e][0]-W*a)
        s=np.abs(pred_omega[e][1]-W*(1.0-a))
        b=r/(r+s+W)
        d=s/(r+s+W)
        u=W/(r+s+W)
        # b = (pred_omega[0]-1.0)/(pred_omega[0]+pred_omega[1])
        # d = (pred_omega[1]-1.0)/(pred_omega[0]+pred_omega[1])
        # u = W/(pred_omega[0]+pred_omega[1])
        Opinions[e]=(b,d,u)
    return Opinions

def calc_Omega_from_Obs3(Obs,E_X):
    W=2.0
    a=0.5
    T = len(Obs.values()[0])
    Omega = {}
    Opinions={}
    for e in E_X:
        pos_evidence=np.sum(Obs[e])* 1.0
        neg_evidence=T - pos_evidence
        b = pos_evidence/(pos_evidence+neg_evidence+W)
        d = neg_evidence / (pos_evidence + neg_evidence + W)
        u = W/(pos_evidence + neg_evidence + W)
        alpha=W*b/u + W*a
        beta= W*d/u + W*(1-a)
        Opinions[e]=(b,d,u)
        Omega[e] = (alpha,beta)
    return Opinions


def calculate_measures2(true_omega_x, pred_belief_x, X):
    bs = []
    for e in X:
        e = tuple(e)
        b1, d1, u1, a1 = beta_to_opinion(true_omega_x[e][0], true_omega_x[e][1])
        b2 = pred_belief_x[e]
        bs.append(np.abs(b1 - b2))
    b_mse = np.mean(bs)
    prob_true_X = {
    tuple(e): (true_omega_x[tuple(e)][0] * 1.0) / (true_omega_x[tuple(e)][0] + true_omega_x[tuple(e)][1]) + 0.0001 for e
    in X}
    prob_mse = np.mean([np.abs(pred_belief_x[tuple(e)] - prob_true_X[tuple(e)]) for e in X])
    return b_mse, prob_mse


def calculate_measures(true_omega_x, pred_omega_x, X):
    W = 2.0
    bs = []
    ds = []
    for e in X:
        b1, d1, u1, a1 = beta_to_opinion(true_omega_x[e][0], true_omega_x[e][1])
        b2, d2, u2, a2 = beta_to_opinion(pred_omega_x[e][0], pred_omega_x[e][1])
        bs.append(np.abs(b1 - b2))
        ds.append(np.abs(d1 - d2))
    b_mse = np.mean(bs)
    d_mse = np.mean(ds)
    alpha_mse = np.mean([np.abs(true_omega_x[e][0] - pred_omega_x[e][0]) for e in X])
    beta_mse = np.mean([np.abs(true_omega_x[e][1] - pred_omega_x[e][1]) for e in X])
    u_true_X = {e: np.abs((W * 1.0) / (true_omega_x[e][0] + true_omega_x[e][1] + W)) for e in X}
    u_pred_X = {e: np.abs((W * 1.0) / (pred_omega_x[e][0] + pred_omega_x[e][1] + W)) for e in X}
    u_mse = np.mean([np.abs(u_pred_X[e] - u_true_X[e]) for e in X])
    u_relative_mse = np.mean([abs(u_pred_X[e] - u_true_X[e]) / u_true_X[e] for e in X])
    prob_true_X = {e: (true_omega_x[e][0] * 1.0) / (true_omega_x[e][0] + true_omega_x[e][1]) + 0.0001 for e in X}
    prob_pred_X = {e: (pred_omega_x[e][0] * 1.0) / (pred_omega_x[e][0] + pred_omega_x[e][1]) for e in X}
    prob_mse = np.mean([np.abs(prob_pred_X[e] - prob_true_X[e]) for e in X])
    prob_relative_mse = np.mean([np.abs(prob_pred_X[e] - prob_true_X[e]) / prob_true_X[e] for e in X])
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
    return alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested


def save_results(dataset, weekday, hour, refspeed, window, percent, alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse,
                 prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested, running_time):
    result = {'dataset': dataset, 'weekday': weekday, 'hour': hour, 'refspeed': refspeed, 'time_window': window,
              'test_ratio': percent, 'alpha_mse': alpha_mse, 'beta_mse': beta_mse, 'prob_mse': prob_mse, 'u_mse': u_mse,
              'b_mse': b_mse, 'd_mse': d_mse, 'prob_relative_mse': prob_relative_mse, 'u_relative_mse': u_relative_mse,
              'accuracy': accuracy, 'recall_congested': recall_congested, 'recall_uncongested': recall_uncongested,
              'running_time': running_time}
    output_file = open('psl_results.json', 'a')
    output_file.write(json.dumps(result) + '\n')
    output_file.close()


def save_results2(graph_size,ratio,real_i,window,ratio_conflict, percent,accuracy, alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, running_time):
    result_ = {'network_size': graph_size, 'positive_ratio': ratio, "realization": real_i,
               'sample_size': 1, 'T': window, 'ratio_conflict': ratio_conflict,
               'test_ratio': percent, 'acc': (accuracy, 0),
               'alpha_mse': (alpha_mse, 0), 'beta_mse': (beta_mse, 0),
               'u_mse': (u_mse, 0), 'b_mse': (b_mse, 0), 'd_mse': (d_mse, 0),
               'prob_mse': (prob_mse, 0), 'runtime': running_time}
    output_file = open('./output/test/psl_results-server-June7-5000.json', 'a')
    output_file.write(json.dumps(result_) + '\n')
    output_file.close()

def sliding_window_extract(Obs, start_t, window_size = 1):
    sw_Obs = {}
    sw_Omega = {}
    for e, obs in Obs.items():
        sw_Obs[e] = [Obs[e][t] for t in range(start_t-window_size, start_t)]
        n = np.sum(sw_Obs[e])
        sw_Omega[e] = (n+1,window_size-n+1)
    return sw_Omega, sw_Obs

def epinion_resutls():
    data_root = "/network/rit/lab/ceashpc/adil/"
    count=0
    for adv_type in ["random_flip", "random_noise", "random_pgd","random_pgd_csl"][3:]:
        for graph_size in [5000,1000,10000,47676][:1]:
            result_folder = data_root + "/result_adv_csl/" + str(graph_size) + "/" + adv_type + "/"
            running_time_dict = read_running_time(result_folder+'running_time.json')
            # exfiles = [file for file in os.listdir(result_folder) if file.endswith(".txt")]
            num_consumers = 30  # We only use 5 cores.
            tasks = multiprocessing.Queue()
            results = multiprocessing.Queue()
            print 'Creating %d consumers' % num_consumers
            consumers = [Consumer(tasks, results)
                         for i in range(num_consumers)]
            for w in consumers:
                w.start()
            num_job=0.0
            for T in [8,9,10,11][2:3]:
                for ratio in [0.2][:1]:
                    for swap_ratio in [0.0]:
                        for percent in [0.1,0.2,0.3,0.4,0.5][:]:
                            for gamma in [0.0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.20, 0.25][:]:  # 11
                                for real_i in range(1):
                                    tasks.put(Task_inference2(graph_size, T, ratio, percent, swap_ratio, gamma, real_i,running_time_dict,result_folder,adv_type))
                                    num_job+=1.0

            for i in xrange(num_consumers):
                tasks.put(None)
            op_results={}
            while num_job:
                # True_Opinions, pred_opinions, test_ratio,ratio_conflict,T=results.get()
                results.get()
                num_job-=1.0
                # op_results["psl-"+str(test_ratio)+"-"+str(gamma)+"-"+str(T)]=(True_Opinions,pred_opinions)
                print num_job
            # outfp = open("../output/test/results-server-Nov12-5000-psl-0.2-tr-ConfRatio-val-T246.pkl", 'a')
            # pickle.dump(op_results, outfp)
            # outfp.close()


    print "Done"

if __name__ == '__main__':
    epinion_resutls()

