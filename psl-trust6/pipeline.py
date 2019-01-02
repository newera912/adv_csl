import pickle
import random
import numpy as np
import subprocess
import re
import time
import json


def current_obs_extract(Obs, start_t):
    current_Obs = {}
    for e, obs in Obs.items():
        current_Obs[e] = Obs[e][start_t-1]
    return current_Obs


def sliding_window_extract(Obs, start_t, window_size = 1):
    sw_Obs = {}
    sw_Omega = {}
    for e, obs in Obs.items():
        sw_Obs[e] = [Obs[e][t] for t in range(start_t-window_size, start_t)]
        n = np.sum(sw_Obs[e])
        sw_Omega[e] = (n+1,window_size - n + 1)
    return sw_Omega, sw_Obs


def generate_data(weekday,hour,refspeed,window,percent):
    '''
    generate evidence data to feed the psl  
    '''
    f = open('../raw_data/raw_network_dc_weekday_'+str(weekday)+'_hour_'+str(hour)+'_refspeed_'+str(refspeed)+'.pkl','rb')
    [V,E,Obs] = pickle.load(f)
    f.close()

    sizeE = len(E)
    #E_X = random.sample(E,int(round(sizeE*percent)))
    
    #'''#this sampling is to avoid isolated testing edge
    E_X = []
    size = int(round(sizeE*percent))
    print size
    while len(E_X)< size:
        all_E = [tuple(e) for e in np.random.permutation(E)]
        for e in all_E:
            source, target = e
            for v1,v2 in E:
                if v2 == source:
                    if e not in E_X:
                        E_X.append(e)
                        break
            break
    #'''
    print 'len(E_X)',len(E_X)
    
    E_X_file = open('results/E_X.json','a')
    E_X_dict = {}
    key = str(weekday)+'_'+str(hour)+'_' +str(refspeed)+'_'+str(window)+'_'+str(percent)
    E_X_dict[key] = E_X
    E_X_file.write(json.dumps(E_X_dict)+'\n')
    E_X_file.close()



    adj_obs = open('../data/adjacent_obs.txt','w')
    edges = []
    for e in E:
        source , target = e
        if e in edges:
            print 'ERROR2'
            print e
            continue
        for v1,v2 in edges:
            if source == v1 or source == v2:
                adj_obs.write(str(source)+'_'+str(target)+'\t'+str(v1)+'_'+str(v2)+'\n')
            elif target == v1 or target == v2:
                adj_obs.write(str(source)+'_'+str(target)+'\t'+str(v1)+'_'+str(v2)+'\n')
        edges.append(e)
    adj_obs.close()   
    

    sw_Omega, sw_Obs = sliding_window_extract(Obs,window)
    current_Obs = current_obs_extract(Obs,window)
    print "sliding window: {0} to {1}".format(window-1,window-1)
    print "Time : {0} will be inferred".format(window-1)
    conj_obs = open('../data/conjested_obs.txt','w')
    nonconj_obs = open('../data/nonconjested_obs.txt','w')
    conj_targets = open('../data/conjested_targets.txt','w')
    #nonconj_targets = open('../data/nonconjested_targets.txt','w')
    conj_truth = open('../data/conjested_truth.txt','w')
    #nonconj_truth = open('../data/nonconjested_truth.txt','w')
    for e in current_Obs:
        if e in E_X:
            source,target = e
            conj_targets.write(str(source)+'_'+str(target)+'\n')
            #nonconj_targets.write(str(source)+'_'+str(target)+'\n')
            
            conj = current_Obs[e]
            if conj == 1:
                conj_truth.write(str(source)+'_'+str(target)+'\t'+'1'+'\n')
                #nonconj_truth.write(str(source)+'_'+str(target)+'\t'+'0'+'\n')
            elif conj == 0:
                conj_truth.write(str(source)+'_'+str(target)+'\t'+'0'+'\n')
                #nonconj_truth.write(str(source)+'_'+str(target)+'\t'+'1'+'\n')
        else:
            conj = current_Obs[e]
            source,target = e
            if conj == 1:
                conj_obs.write(str(source)+'_'+str(target)+'\n')
            elif conj == 0:
                nonconj_obs.write(str(source)+'_'+str(target)+'\n')
                ## shall we need to consider the ground rule of nonconjested
            else:
                print conj
                raise Exception('no obs error')
    conj_obs.close()
    nonconj_obs.close()
    conj_targets.close()
    #nonconj_targets.close()
    conj_truth.close()
    #nonconj_truth.close()

    return sw_Omega,E_X



def result_analysis(sw_Omega,E_X):
    f = open('output/default/conjested_infer.txt','r')
    lines = f.readlines()
    Omega_X = {}
    for line in lines[1:-1]:
        fields = re.split('\'|\[|\]',line) 
        edge = fields[1].split('_')
        source = int(edge[0])
        target = int(edge[1])
        e = (source,target)
        pred = float(fields[3])
        #print e,pred
        if e not in E_X:
            continue
        else:
            alpha = pred
            if e not in Omega_X:
                if alpha == 0:
                    Omega_X[e] = (1,2)
                elif alpha == 1:
                    Omega_X[e] = (2,1)
                else:
                    Omega_X[e] = (alpha,1.0-alpha)
    count = 0.0
    for e in E_X:
        if e not in Omega_X:
            count += 1
            Omega_X[e] = (1.0,1.0)
    #print 'Not Predicted', count
    alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested = calculate_measures(sw_Omega,Omega_X,E_X) 
    return alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested
 

def beta_to_opinion(alpha,beta,W=2.0,a=0.5):
    '''
    compute opinion based on hyperparameters of beta distribution
    '''
    b = (alpha - W*a)/float(alpha+beta)
    d = (beta - W*(1-a))/float(alpha+beta)
    u = (W)/float(alpha+beta)
    return [b,d,u,a]


def calculate_measures(true_omega_x, pred_omega_x, X):
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
    u_mse = np.mean([np.abs(u_pred_X[e] - u_true_X[e]) for e in X])
    u_relative_mse = np.mean([abs(u_pred_X[e] - u_true_X[e])/u_true_X[e] for e in X])
    prob_true_X = {e: (true_omega_x[e][0]*1.0)/(true_omega_x[e][0] + true_omega_x[e][1])+0.0001 for e in X}
    prob_pred_X = {e: (pred_omega_x[e][0]*1.0)/(pred_omega_x[e][0] + pred_omega_x[e][1]) for e in X}
    prob_mse = np.mean([np.abs(prob_pred_X[e] - prob_true_X[e]) for e in X])
    prob_relative_mse = np.mean([np.abs(prob_pred_X[e] - prob_true_X[e])/prob_true_X[e] for e in X])
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



def save_results(dataset,weekday,hour,refspeed,window,percent,alpha_mse,beta_mse,prob_mse,u_mse,b_mse,d_mse,prob_relative_mse,u_relative_mse,accuracy,recall_congested,recall_uncongested,running_time):
    result = {'dataset':dataset, 'weekday':weekday, 'hour':hour, 'refspeed':refspeed, 'time_window':window, 'test_ratio': percent, 'alpha_mse':alpha_mse,'beta_mse':beta_mse,'prob_mse':prob_mse, 'u_mse':u_mse,'b_mse':b_mse,'d_mse':d_mse, 'prob_relative_mse':prob_relative_mse, 'u_relative_mse':u_relative_mse, 'accuracy':accuracy, 'recall_congested':recall_congested,'recall_uncongested':recall_uncongested, 'running_time':running_time}
    output_file = open('psl_result.json','a')
    output_file.write(json.dumps(result)+'\n')
    output_file.close()





def pipeline():
    for dataset in ['dc']:
        for weekday in range(4,5):
            for hour in range(6,22):
                for refspeed  in [0.8]:
                    for window in range(1,44):
                        for percent in [0.1,0.2,0.3,0.4,0.5]:
                            running_start_time = time.time()
                            print weekday,hour,refspeed,window,percent
                            sw_Omega, E_X = generate_data(weekday,hour,refspeed,window,percent)
                            proc = subprocess.Popen(["./run.sh"])
                            proc.communicate()
                            proc.wait()
                            result_file = str(weekday)+'_'+str(hour)+'_' +str(refspeed)+'_'+str(window)+'_'+str(percent)
                            proc = subprocess.Popen(["cp","output/default/conjested_infer.txt","results/"+result_file])
                            proc.communicate()
                            proc.wait()
                            running_end_time = time.time()
                            running_time = running_end_time - running_start_time
                            alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested = result_analysis(sw_Omega,E_X)
                            save_results(dataset,weekday,hour,refspeed,window,percent,alpha_mse,beta_mse,prob_mse,u_mse,b_mse,d_mse,prob_relative_mse,u_relative_mse,accuracy,recall_congested,recall_uncongested,running_time)






if __name__ == '__main__':
    pipeline()
