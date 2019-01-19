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


def generate_data(Obs, E, E_X,gamma):
    adj_obs = open('../data/adjacent_obs.txt', 'w')
    edges = {}
    for e in E:
        if e[0] < e[1]:
            source = e[0]
            target = e[1]
        else:
            source = e[1]
            target = e[0]
        # source = e[0]
        # target = e[1]
        e = (source, target)
        if edges.has_key(e):
            # print 'ERROR2'
            # print e
            continue
        adj_obs.write(str(source) + '\t' + str(target) + '\n')

        edges[e] = 1
    print len(edges)
    adj_obs.close()

    current_Obs = Obs
    # print "sliding window: {0} to {1}".format(window-5,window-1)
    # print "Time : {0} will be inferred".format(window-1)
    sybils_obs = open('../data/sybils_obs.txt', 'w')
    benign_obs = open('../data/benign_obs.txt', 'w')
    sybils_targets = open('../data/sybils_targets.txt', 'w')
    # nonconj_targets = open('../data/nonconjested_targets.txt','w')
    sybils_truth = open('../data/sybils_truth.txt', 'w')
    # nonconj_truth = open('../data/nonconjested_truth.txt','w')
    for v in current_Obs.keys():
        if E_X.has_key(v):
            sybils_targets.write(str(v) + '\n')
            # nonconj_targets.write(str(source)+'_'+str(target)+'\n')

            benign = current_Obs[v]
            if benign >= 0.5:
                sybils_truth.write(str(v) + '\t' + '1' + '\n')
                # nonconj_truth.write(str(source)+'_'+str(target)+'\t'+'0'+'\n')
            elif benign <= 0.5:
                sybils_truth.write(str(v) + '\t' + '0' + '\n')
                # nonconj_truth.write(str(source)+'_'+str(target)+'\t'+'1'+'\n')
        else:
            benign = current_Obs[v]
            if benign >0.5:
                sybils_obs.write(str(v) + '\n')
            elif benign <=0.5:
                benign_obs.write(str(v) + '\n')
            else:
                print benign
                raise Exception('no obs error')
    sybils_obs.close()
    benign_obs.close()
    sybils_targets.close()
    # nonconj_targets.close()
    sybils_truth.close()
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
    # sybils2
    data_root = "/network/rit/lab/ceashpc/adil/"

    report_stat = False
    count = 0
    realizations = 1
    # methods = ["sl", "csl", "csl-conflict-1", "csl-conflict-2", "base1", "base2", "base3"][:1]

    for test_ratio in [0.3,0.1, 0.2, 0.4, 0.5][:]:
        for adv_type in ["random_noise","random_pgd","random_pgd_csl","random_pgd_gcn_vae"][:]:
            for attack_edge in [1000, 5000, 10000, 15000, 20000][2:3]:
                result_folder = data_root + "/result_adv_csl/facebook/" + adv_type + "/"
                if not os.path.exists(result_folder):
                    os.makedirs(result_folder)
                exfiles = {file: 1 for file in os.listdir(result_folder) if file.endswith(".txt")}
                for T in [10][:]:
                    for swap_ratio in [0.00, 0.01, 0.02, 0.05][1:2]:
                        for gamma in [0.0, 0.01, 0.03, 0.05, 0.07,0.09,0.2,0.3,0.4,0.5][:]:  # 11
                            for real_i in range(realizations)[:1]:
                                count += 1.0
                                f = data_root + "data/adv_csl/Jan2/" + adv_type + "/facebook/facebook-attackedges-{}-T-{}-testratio-{}-swap_ratio-{}-gamma-{}-realization-{}-data-X.pkl".format(
                                    attack_edge, T, test_ratio, swap_ratio, gamma, real_i)
                                print(f)
                                pkl_file = open(f, 'rb')
                                # V, E, Obs, E_X, X_b
                                [_, E, Obs, E_X, _] = pickle.load(pkl_file)
                                pkl_file.close()
                                # E_X = {e: 1 for e in E_X}
                                T = len(Obs.values()[0])
                                m_idx = int(round(T / 2.0))
                                # for window in range(m_idx - 5,m_idx + 7):
                                for window in range(T)[:3]:
                                    running_start_time = time.time()
                                    t_Obs = {v: v_Obs[window] for v, v_Obs in Obs.items()}
                                    result_file = str(attack_edge) + '_' + str(
                                        test_ratio) + '_' + str(
                                        swap_ratio) + '_' + str(gamma) + '_' + str(window) + '_' + str(real_i) + '.txt'
                                    if exfiles.has_key(result_file):
                                        print "exists", result_file
                                        continue
                                    print ">>>>", count, "-th ", attack_edge, real_i, T, window, test_ratio, gamma

                                    generate_data(t_Obs, E, E_X,gamma)
                                    proc = subprocess.Popen(["./run.sh"])
                                    proc.communicate()
                                    proc.wait()
                                    proc = subprocess.Popen(
                                        ["cp", "output/default/sybils_infer.txt", result_folder + result_file])
                                    proc.communicate()
                                    proc.wait()
                                    running_end_time = time.time()
                                    running_time = running_end_time - running_start_time
                                    key = str(attack_edge) + '_' + str(
                                        test_ratio) + '_' + str(
                                        swap_ratio) + '_' + str(gamma) + '_' + str(window) + '_' + str(
                                        real_i)
                                    r_dict = {}
                                    r_dict[key] = running_time
                                    with open(result_folder + '/running_time.json', 'a') as op:
                                        op.write(json.dumps(r_dict) + '\n')
                                    count += 1


if __name__ == '__main__':
    pipeline()
