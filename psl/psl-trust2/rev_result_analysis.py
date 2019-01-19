import json
import pickle
import numpy as np
import re

def read_raw_data(graph_size,ratio,swapratio,real_i):
    f = open("trust/easy/raw_data/nodes-{}-rate-{}-swaprate-{}-realization-{}-data.pkl".format(graph_size, ratio, swapratio, real_i),'rb')
    [V,E,Obs] = pickle.load(f)
    f.close()
    return Obs


def sliding_window_extract(Obs, start_t, T):
    sw_Obs = {}
    sw_Omega = {}
    for e, obs in Obs.items():
        sw_Obs[e] = [Obs[e][t] for t in range(start_t-T, start_t)]
        n = np.sum(sw_Obs[e])
        sw_Omega[e] = (n+1, T-n+1)
    return sw_Omega, sw_Obs


def read_E_X_dict(graph_size):
    E_X_dict = {}
    E_X_f = open('trust/easy/groovy/results/E_X.json','r')
    for line in E_X_f:
        [(key,value)] = json.loads(line).items()
        if key not in E_X_dict:
            E_X_dict[key] = value
    return E_X_dict


def read_running_time():
    running_time_dict = {}
    f = open('trust/easy/groovy/results/running_time.json','r')
    for line in f:
        result = json.loads(line)
        key = result.keys()[0]
        print key
        running_time = result[key]
        if key not in running_time_dict:
            running_time_dict[key] = running_time
    return running_time_dict


def result_analysis(E_X_dict,sw_Omega,graph_size,ratio,swapratio,real_i,window,percent):
    key = str(graph_size)+'_'+str(ratio)+'_'+str(real_i)+'_'+str(window)+'_'+str(percent)
    print key
    E_X = E_X_dict[key]
    key = str(graph_size)+'_'+str(ratio)+'_'+str(swapratio)+'_'+str(real_i)+'_'+str(window)+'_'+str(percent)
    f = open('trust/easy/groovy/results/'+key,'r')


    lines = f.readlines()
    Omega_X = {}
    pred_belief = {}
    for line in lines[1:-1]:
        fields = re.split('\'|\[|\]',line) 
        print fields
        source = int(fields[1])
        target = int(fields[3])
        e = (source,target)
        pred = float(fields[5])
        if list(e) not in E_X:
            continue
        else:
            alpha = pred
            if e not in pred_belief:
                pred_belief[e] = alpha

    for e in E_X:
        if tuple(e) not in pred_belief:
            pred_belief[tuple(e)] = 0
    print pred_belief
    b_mse, prob_mse = calculate_measures2(sw_Omega,pred_belief,E_X)
    return b_mse, prob_mse
    
 

def trim(val):
    if val < 0:
        return 0
    elif val > 1:
        return 1
    else:
        return val


def beta_to_opinion(alpha,beta,W=2.0,a=0.5):
    '''
    compute opinion based on hyperparameters of beta distribution
    '''
    b = trim((alpha - W*a)/float(alpha+beta))
    d = trim((beta - W*(1-a))/float(alpha+beta))
    u = trim((W)/float(alpha+beta+W))
    return [b,d,u,a]


def calculate_measures2(true_omega_x, pred_belief_x, X):
    print pred_belief_x
    bs = []
    for e in X:
        e = tuple(e)
        b1,d1,u1,a1 = beta_to_opinion(true_omega_x[e][0], true_omega_x[e][1])
        b2 = pred_belief_x[e]
        bs.append(np.abs(b1-b2))
    b_mse = np.mean(bs)
    prob_true_X = {tuple(e): (true_omega_x[tuple(e)][0]*1.0)/(true_omega_x[tuple(e)][0] + true_omega_x[tuple(e)][1])+0.0001 for e in X}
    prob_mse = np.mean([np.abs(pred_belief_x[tuple(e)] - prob_true_X[tuple(e)]) for e in X])
    return b_mse, prob_mse



def save_results2(graph_size,ratio,swapratio,real_i,window,percent,prob_mse,b_mse,running_time,T):
    result = {'network_size':graph_size, 'positive_ratio':ratio, 'swapratio':swapratio, 'realization':real_i, 'test_ratio': percent, 'prob_mse':prob_mse, 'b_mse':b_mse, 'running_time':running_time, 'T':T}
    output_file = open('psl_results.json','a')
    output_file.write(json.dumps(result)+'\n')
    output_file.close()


if __name__ == '__main__':
    running_time_dict = read_running_time()
    for T in [6] :
        for graph_size in [47676]:
            E_X_dict = read_E_X_dict(graph_size)
            for ratio in [0.2]:
                for swapratio in [0.05]:
                    for real_i in range(1):
                        window = T
                        Obs = read_raw_data(graph_size,ratio,swapratio,real_i)
                        sw_Omega, sw_Obs = sliding_window_extract(Obs,window,T)
                        for percent in [0.2]:
                            key = str(graph_size)+'_'+str(ratio)+'_' +str(swapratio)+'_'+str(real_i)+'_'+str(window)+'_'+str(percent)
                            print 'key',key
                            running_time = running_time_dict[key]
                            #alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested = result_analysis(E_X_dict, sw_Omega, weekday, hour, refspeed, window, percent)
                            #save_results(dataset,weekday,hour,refspeed,window,percent,alpha_mse,beta_mse,prob_mse,u_mse,b_mse,d_mse,prob_relative_mse,u_relative_mse,accuracy,recall_congested,recall_uncongested,running_time)
                            b_mse, prob_mse = result_analysis(E_X_dict,sw_Omega,graph_size,ratio,swapratio,real_i,window,percent)
                            save_results2(graph_size,ratio,swapratio,real_i,window,percent,prob_mse,b_mse,running_time,T)
