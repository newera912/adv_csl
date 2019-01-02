import pickle
import random
import numpy as np
import subprocess
import re, os
import time
import json


def current_obs_extract(Obs, start_t):
    current_Obs = {}
    for e, obs in Obs.items():
        current_Obs[e] = Obs[e][start_t - 1]
    return current_Obs


def sliding_window_extract(Obs, start_t, window_size=5):
    sw_Obs = {}
    sw_Omega = {}
    for e, obs in Obs.items():
        sw_Obs[e] = [Obs[e][t] for t in range(start_t - window_size, start_t)]
        n = np.sum(sw_Obs[e])
        sw_Omega[e] = (n + 1, window_size - n + 1)
    return sw_Omega, sw_Obs


def generate_data(Obs, E, E_X):
    sizeE = len(E)

    adj_obs = open('../data/adjacent_obs.txt', 'w')
    edges = {}
    for e in E:
        source, target = e
        if edges.has_key(e):
            print 'ERROR2'
            print e
            continue
        for v1, v2 in edges:
            if source == v2:
                adj_obs.write(str(v1) + '_' + str(v2) + '\t' + str(source) + '_' + str(target) + '\n')
            elif target == v1:
                adj_obs.write(str(source) + '_' + str(target) + '\t' + str(v1) + '_' + str(v2) + '\n')
        # for v1,v2 in edges:
        #     if source == v1 or source == v2:
        #         adj_obs.write(str(source)+'_'+str(target)+'\t'+str(v1)+'_'+str(v2)+'\n')
        #     elif target == v1 or target == v2:
        #         adj_obs.write(str(source)+'_'+str(target)+'\t'+str(v1)+'_'+str(v2)+'\n')
        edges[e] = 1
    adj_obs.close()

    current_Obs = Obs
    # print "sliding window: {0} to {1}".format(window-5,window-1)
    # print "Time : {0} will be inferred".format(window-1)
    conj_obs = open('../data/conjested_obs.txt', 'w')
    nonconj_obs = open('../data/nonconjested_obs.txt', 'w')
    conj_targets = open('../data/conjested_targets.txt', 'w')
    # nonconj_targets = open('../data/nonconjested_targets.txt','w')
    conj_truth = open('../data/conjested_truth.txt', 'w')
    # nonconj_truth = open('../data/nonconjested_truth.txt','w')
    for e in current_Obs:
        if e in E_X:
            source, target = e
            conj_targets.write(str(source) + '_' + str(target) + '\n')
            # nonconj_targets.write(str(source)+'_'+str(target)+'\n')

            conj = current_Obs[e][0]
            if conj == 1:
                conj_truth.write(str(source) + '_' + str(target) + '\t' + '1' + '\n')
                # nonconj_truth.write(str(source)+'_'+str(target)+'\t'+'0'+'\n')
            elif conj == 0:
                conj_truth.write(str(source) + '_' + str(target) + '\t' + '0' + '\n')
                # nonconj_truth.write(str(source)+'_'+str(target)+'\t'+'1'+'\n')
        else:
            conj = current_Obs[e][0]
            source, target = e
            if conj == 1:
                conj_obs.write(str(source) + '_' + str(target) + '\n')
            elif conj == 0:
                nonconj_obs.write(str(source) + '_' + str(target) + '\n')
                ## shall we need to consider the ground rule of nonconjested
            else:
                print conj
                raise Exception('no obs error')
    conj_obs.close()
    nonconj_obs.close()
    conj_targets.close()
    # nonconj_targets.close()
    conj_truth.close()
    # nonconj_truth.close()

    return


def result_analysis(sw_Omega, E_X):
    f = open('output/default/conjested_infer.txt', 'r')
    lines = f.readlines()
    Omega_X = {}
    for line in lines[1:-1]:
        fields = re.split('\'|\[|\]', line)
        edge = fields[1].split('_')
        source = int(edge[0])
        target = int(edge[1])
        e = (source, target)
        pred = float(fields[3])
        # print e,pred
        if e not in E_X:
            continue
        else:
            alpha = pred
            if e not in Omega_X:
                if alpha == 0:
                    Omega_X[e] = (1, 2)
                elif alpha == 1:
                    Omega_X[e] = (2, 1)
                else:
                    Omega_X[e] = (alpha, 1.0 - alpha)
    count = 0.0
    for e in E_X:
        if e not in Omega_X:
            count += 1
            Omega_X[e] = (1.0, 1.0)
    # print 'Not Predicted', count
    prob_mse, u_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested = calculate_measures(
        sw_Omega, Omega_X, E_X)
    return prob_mse, u_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested


def save_results(dataset, weekday, hour, refspeed, window, percent, prob_mse, u_mse, prob_relative_mse, u_relative_mse,
                 accuracy, recall_congested, recall_uncongested, running_time):
    result = {'dataset': dataset, 'weekday': weekday, 'hour': hour, 'refspeed': refspeed, 'time_window': window,
              'test_ratio': percent, 'prob_mse': prob_mse, 'u_mse': u_mse, 'prob_relative_mse': prob_relative_mse,
              'u_relative_mse': u_relative_mse, 'accuracy': accuracy, 'recall_congested': recall_congested,
              'recall_uncongested': recall_uncongested, 'running_time': running_time}
    output_file = open('psl_result.json', 'a')
    output_file.write(json.dumps(result) + '\n')
    output_file.close()


def calculate_measures(true_omega_x, pred_omega_x, X):
    W = 1.0
    u_true_X = {e: np.abs((W * 1.0) / (true_omega_x[e][0] + true_omega_x[e][1])) for e in X}
    u_pred_X = {e: np.abs((W * 1.0) / (pred_omega_x[e][0] + pred_omega_x[e][1])) for e in X}
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
    return prob_mse, u_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested


def pipeline():
    # pipeline 1
    data_root = "/network/rit/lab/ceashpc/adil/data/csl-data/Dec10/"   #Sep18 #June25/"
    result_folder = "/network/rit/lab/ceashpc/adil/results-traffic/"
    realizations = 1
    ref_pers = [0.6, 0.7, 0.8]
    datasets = ['philly', 'dc']
    count = 0
    for dataset in datasets[1:]:
        exfiles = {file: 1 for file in os.listdir(result_folder) if file.endswith(".txt")}
        for ref_ratio in ref_pers[:]:
            dataroot = data_root + dataset + "/"
            # if not os.path.exists(dataroot):
            #     os.makedirs(dataroot)
            for weekday in range(5)[:]:
                for hour in range(8, 22)[:]:
                    for test_ratio in [0.1, 0.2, 0.3, 0.4, 0.5][1:2]:
                        for ratio_conflict in [0.0, 0.1, 0.2, 0.3, 0.4][3:4]:
                            for real_i in range(realizations)[:1]:
                                f = dataroot + '/network_{}_weekday_{}_hour_{}_refspeed_{}-testratio-{}-confictratio-{}-realization-{}.pkl'.format(
                                    dataset, weekday, hour, ref_ratio, test_ratio, ratio_conflict, real_i)
                                pkl_file = open(f, 'rb')
                                [_, E, Obs, E_X, _] = pickle.load(pkl_file)
                                pkl_file.close()
                                E_X = {e: 1 for e in E_X}
                                T = len(Obs[E[0]])
                                m_idx = int(round(T / 2.0))
                                # for window in range(m_idx - 5, m_idx + 7):
                                for window in range(T):
                                    running_start_time = time.time()
                                    t_Obs = {e: e_Obs[window:window + 1] for e, e_Obs in Obs.items()}
                                    result_file = str(dataset) + '_' + str(weekday) + '_' + str(hour) + '_' + str(
                                        ref_ratio) + '_' + str(
                                        test_ratio) + '_' + str(ratio_conflict) + '_' + str(window) + '_' + str(
                                        real_i) + '-T43-1210.txt'
                                    if exfiles.has_key(result_file):
                                        continue

                                    print ">>>>", count, "-th ", dataset, ref_ratio, weekday, hour, real_i, T, window, test_ratio, ratio_conflict

                                    generate_data(t_Obs, E, E_X)
                                    proc = subprocess.Popen(["./run.sh"])
                                    proc.communicate()
                                    proc.wait()
                                    proc = subprocess.Popen(
                                        ["cp", "output/default/conjested_infer.txt", result_folder + result_file])
                                    proc.communicate()
                                    proc.wait()
                                    running_end_time = time.time()
                                    running_time = running_end_time - running_start_time
                                    key = str(dataset) + '_' + str(weekday) + '_' + str(hour) + '_' + str(
                                        ref_ratio) + '_' + str(
                                        test_ratio) + '_' + str(ratio_conflict) + '_' + str(window) + '_' + str(real_i)
                                    r_dict = {}
                                    r_dict[key] = running_time
                                    with open(result_folder + '/running_time.json', 'a') as op:
                                        op.write(json.dumps(r_dict) + '\n')
                                    count += 1


if __name__ == '__main__':
    pipeline()
