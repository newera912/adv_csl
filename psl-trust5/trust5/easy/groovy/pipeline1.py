import pickle
import random
import numpy as np
import subprocess
import re
import time
import json, os


def current_obs_extract(Obs, start_t):
    current_Obs = {}
    for e, obs in Obs.items():
        current_Obs[e] = Obs[e][start_t]
    return current_Obs


def sliding_window_extract(Obs, start_t, window_size=1):
    sw_Obs = {}
    sw_Omega = {}
    for e, obs in Obs.items():
        sw_Obs[e] = [Obs[e][t] for t in range(start_t - window_size, start_t)]
        n = np.sum(sw_Obs[e])
        sw_Omega[e] = (n + 1, window_size - n + 1)
    return sw_Omega, sw_Obs


def generate_data(Obs, E, E_X, T, window):
    sizeE = len(E)
    #     E_X = random.sample(E,int(round(sizeE*percent)))
    # '''#this sampling is to avoid isolated testing edge

    print 'len(E_X)', len(E_X)

    adj_obs = open('../data/adjacent_obs.txt', 'w')
    for e in E:
        source, target = e
        adj_obs.write(str(source) + '\t' + str(target) + '\n')
    adj_obs.close()

    # sw_Omega, sw_Obs = sliding_window_extract(Obs, window)
    current_Obs = current_obs_extract(Obs, window)
    print "sliding window: {0} to {1}".format(window - 1, window - 1)
    print "Time : {0} will be inferred".format(window - 1)
    trust_obs = open('../data/T_obs.txt', 'w')
    trust_targets = open('../data/T_targets.txt', 'w')
    trust_truth = open('../data/T_truth.txt', 'w')
    for e in current_Obs:
        if E_X.has_key(e):
            source, target = e
            trust_targets.write(str(source) + '\t' + str(target) + '\n')

            trust = current_Obs[e]
            if trust == 1:
                trust_truth.write(str(source) + '\t' + str(target) + '\t' + '1' + '\n')
                # nonconj_truth.write(str(source)+'_'+str(target)+'\t'+'0'+'\n')
            elif trust == 0:
                trust_truth.write(str(source) + '\t' + str(target) + '\t' + '0' + '\n')
                # nonconj_truth.write(str(source)+'_'+str(target)+'\t'+'1'+'\n')
        else:
            trust = current_Obs[e]
            source, target = e
            if trust == 1:
                trust_obs.write(str(source) + '\t' + str(target) + '\n')

    trust_obs.close()
    trust_targets.close()
    trust_truth.close()


def pipeline():
    # trust5
    data_root = "/network/rit/lab/ceashpc/adil/"
    count = 0
    # with open('results/running_time.json','a') as outfile:
    for graph_size in [1000, 5000, 10000, 47676][1:2]:
        folder = data_root + "results-" + str(
            graph_size) + "/"
        exfiles = {file: 1 for file in os.listdir(folder) if file.endswith(".txt")}
        result_folder = data_root + "results-" + str(graph_size) + "/"
        for T in [2,4,6,8, 9, 10, 11][:3]:
            for ratio in [0.2,0.6, 0.7, 0.8][:1]:
                for swap_ratio in [0.0]:
                    for percent in [0.1, 0.2, 0.3, 0.4, 0.5][:]:
                        for ratio_conflict in [0.0, 0.1, 0.2, 0.3, 0.4][1:]:
                            for real_i in range(1):
                                '''
                                    generate evidence data to feed the psl
                                    '''
                                # nodes-47676-T-10-rate-0.1-testratio-0.1-swaprate-0.0-confictratio-0.0-realization-0-data-X.pkl
                                f = "/network/rit/lab/ceashpc/adil/data/csl-data/Oct10/{}/nodes-{}-T-{}-rate-{}-testratio-{}-swaprate-{}-confictratio-{}-realization-{}-data-X.pkl".format(
                                    graph_size, graph_size, T, ratio, percent, swap_ratio, ratio_conflict, real_i)
                                # f = "/network/rit/lab/ceashpc/adil/data/csl-data/apr21/5000/nodes-{}-T-{}-rate-{}-testratio-{}-swaprate-{}-confictratio-{}-realization-{}-data-X.pkl".format(graph_size, window, ratio, percent, swaprate, ratio_conflict, real_i)

                                print f
                                pkl_file = open(f, 'rb')
                                [_, E, Obs, E_X, _] = pickle.load(pkl_file)  # V, E, Obs, E_X, X_b
                                pkl_file.close()
                                E_X = {e: 1 for e in E_X}

                                for window in range(T):
                                    running_start_time = time.time()
                                    result_file = str(graph_size) + '_' + str(ratio) + '_' + str(
                                        swap_ratio) + '_' + str(
                                        T) + '_' + str(
                                        window) + '_' + str(percent) + '_' + str(ratio_conflict) + '_' + str(
                                        real_i) + '.txt'
                                    if exfiles.has_key(result_file):
                                        continue

                                    print ">>>>", count, "-th ", graph_size, ratio, real_i, T, window, percent, ratio_conflict

                                    generate_data(Obs, E, E_X, T, window)
                                    proc = subprocess.Popen(["./run.sh"])
                                    proc.communicate()
                                    proc.wait()
                                    proc = subprocess.Popen(
                                        ["cp", "output/default/trust_infer.txt", result_folder + result_file])
                                    proc.communicate()
                                    proc.wait()
                                    running_end_time = time.time()
                                    running_time = running_end_time - running_start_time
                                    key = str(graph_size) + '_' + str(ratio) + '_' + str(swap_ratio) + '_' + str(
                                        T) + '_' + str(window) + '_' + str(percent) + '_' + str(
                                        ratio_conflict) + '_' + str(real_i)
                                    r_dict = {}
                                    r_dict[key] = running_time
                                    with open(folder + '/running_time.json', 'a') as op:
                                        op.write(json.dumps(r_dict) + '\n')
                                    count += 1
                                # alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested = result_analysis(sw_Omega,E_X)
                                # save_results(dataset,weekday,hour,refspeed,window,percent,alpha_mse,beta_mse,prob_mse,u_mse,b_mse,d_mse,prob_relative_mse,u_relative_mse,accuracy,recall_congested,recall_uncongested,running_time)
        


def test():
    V = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    E = [(0, 1), (0, 3), (0, 4), (0, 6), (4, 1), (3, 4), (4, 2), (4, 6), (2, 5), (5, 8), (5, 7), (7, 8), (6, 7)]
    Obs = {(0, 1): 0, (0, 3): 1, (0, 4): 1, (0, 6): 1, (4, 1): 0, (3, 4): 1, (4, 2): 0, (4, 6): 1, (2, 5): 0, (5, 8): 0,
           (5, 7): 0, (7, 8): 1, (6, 7): 1}
    E_X = [(0, 6), (0, 1), (0, 4), (5, 8)]

    adj_obs = open('../data/adjacent_obs.txt', 'w')
    for e in E:
        source, target = e
        adj_obs.write(str(source) + '\t' + str(target) + '\n')
    adj_obs.close()

    trust_obs = open('../data/T_obs.txt', 'w')
    trust_targets = open('../data/T_targets.txt', 'w')
    trust_truth = open('../data/T_truth.txt', 'w')
    for e in Obs:
        if e in E_X:
            source, target = e
            trust_targets.write(str(source) + '\t' + str(target) + '\n')

            trust = Obs[e]
            if trust == 1:
                trust_truth.write(str(source) + '\t' + str(target) + '\t' + '1' + '\n')
            elif trust == 0:
                trust_truth.write(str(source) + '\t' + str(target) + '\t' + '0' + '\n')
        else:
            trust = Obs[e]
            source, target = e
            if trust == 1:
                trust_obs.write(str(source) + '\t' + str(target) + '\n')

    trust_obs.close()
    trust_targets.close()
    trust_truth.close()

    proc = subprocess.Popen(["./run.sh"])
    proc.communicate()
    proc.wait()


if __name__ == '__main__':
    pipeline()
    # test()
