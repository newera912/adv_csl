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
        sw_Omega[e] = (n+1,window_size-n+1)
    return sw_Omega, sw_Obs


def generate_data(graph_size,ratio,swaprate,real_i,window,percent):
    '''
    generate evidence data to feed the psl  
    '''
    f = "../raw_data/nodes-{}-T-6-rate-{}-testratio-0.2-swaprate-{}-realization-{}-data-X.pkl".format(graph_size, ratio, swaprate, real_i)
    print f
    pkl_file = open(f,'rb')
    [V,E,Obs,E_X] = pickle.load(pkl_file)
    pkl_file.close()

    
    sizeE = len(E)
    E_X = random.sample(E,int(round(sizeE*percent)))
    #'''#this sampling is to avoid isolated testing edge
    """
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
    """
    print 'len(E_X)',len(E_X)
    E_X_file = open('results/E_X.json','a')
    E_X_dict = {}
    key = str(graph_size)+'_'+str(ratio)+'_' +str(real_i)+'_'+str(window)+'_'+str(percent)
    E_X_dict[key] = E_X
    E_X_file.write(json.dumps(E_X_dict)+'\n')
    E_X_file.close()

    adj_obs = open('../data/adjacent_obs.txt','w')
    for e in E:
        source , target = e
        adj_obs.write(str(source)+'\t'+str(target)+'\n')
    adj_obs.close()   
    

    sw_Omega, sw_Obs = sliding_window_extract(Obs,window)
    current_Obs = current_obs_extract(Obs,window)
    print "sliding window: {0} to {1}".format(window-1,window-1)
    print "Time : {0} will be inferred".format(window-1)
    trust_obs = open('../data/T_obs.txt','w')
    trust_targets = open('../data/T_targets.txt','w')
    trust_truth = open('../data/T_truth.txt','w')
    for e in current_Obs:
        if e in E_X:
            source,target = e
            trust_targets.write(str(source)+'\t'+str(target)+'\n')
            
            trust = current_Obs[e]
            if trust == 1:
                trust_truth.write(str(source)+'\t'+str(target)+'\t'+'1'+'\n')
                #nonconj_truth.write(str(source)+'_'+str(target)+'\t'+'0'+'\n')
            elif trust == 0:
                trust_truth.write(str(source)+'\t'+str(target)+'\t'+'0'+'\n')
                #nonconj_truth.write(str(source)+'_'+str(target)+'\t'+'1'+'\n')
        else:
            trust = current_Obs[e]
            source,target = e
            if trust == 1:
                trust_obs.write(str(source)+'\t'+str(target)+'\n')

    trust_obs.close()
    trust_targets.close()
    trust_truth.close()

    return sw_Omega,E_X

 

def pipeline():
    outfile = open('results/running_time.json','a')
    for graph_size in [2500,7500]:
        for ratio in [0.2]:
            for swap_ratio in [0.05]:
                for real_i in range(1):
                    for window in [6]:
                        for percent in [0.2]:
                            running_start_time = time.time()
                            print graph_size, ratio, real_i, window, percent
                            sw_Omega, E_X = generate_data(graph_size, ratio, swap_ratio,real_i, window, percent)
                            proc = subprocess.Popen(["./run.sh"])
                            proc.communicate()
                            proc.wait()
                            result_file = str(graph_size)+'_'+str(ratio)+'_'+str(swap_ratio)+'_'+str(real_i)+'_'+str(window)+'_'+str(percent)+'.txt'
                            proc = subprocess.Popen(["cp","output/default/trust_infer.txt","results/"+result_file])
                            proc.communicate()
                            proc.wait()
                            running_end_time = time.time()
                            running_time = running_end_time - running_start_time
                            key = str(graph_size)+'_'+str(ratio)+'_'+str(swap_ratio)+'_'+str(real_i)+'_'+str(window)+'_'+str(percent) 
                            r_dict = {}
                            r_dict[key] = running_time
                            outfile.write(json.dumps(r_dict)+'\n')
                            #alpha_mse, beta_mse, prob_mse, u_mse, b_mse, d_mse, prob_relative_mse, u_relative_mse, accuracy, recall_congested, recall_uncongested = result_analysis(sw_Omega,E_X)
                            #save_results(dataset,weekday,hour,refspeed,window,percent,alpha_mse,beta_mse,prob_mse,u_mse,b_mse,d_mse,prob_relative_mse,u_relative_mse,accuracy,recall_congested,recall_uncongested,running_time)
    outfile.close()


def test():
    V = [0,1,2,3,4,5,6,7,8]
    E = [(0,1),(0,3),(0,4),(0,6),(4,1),(3,4),(4,2),(4,6),(2,5),(5,8),(5,7),(7,8),(6,7)]
    Obs = {(0,1):0,(0,3):1,(0,4):1,(0,6):1,(4,1):0,(3,4):1,(4,2):0,(4,6):1,(2,5):0,(5,8):0,(5,7):0,(7,8):1,(6,7):1}
    E_X = [(0,6),(0,1),(0,4),(5,8)]
    
    adj_obs = open('../data/adjacent_obs.txt','w')
    for e in E:
        source , target = e
        adj_obs.write(str(source)+'\t'+str(target)+'\n')
    adj_obs.close()   

    trust_obs = open('../data/T_obs.txt','w')
    trust_targets = open('../data/T_targets.txt','w')
    trust_truth = open('../data/T_truth.txt','w')
    for e in Obs:
        if e in E_X:
            source,target = e
            trust_targets.write(str(source)+'\t'+str(target)+'\n')
            
            trust = Obs[e]
            if trust == 1:
                trust_truth.write(str(source)+'\t'+str(target)+'\t'+'1'+'\n')
            elif trust == 0:
                trust_truth.write(str(source)+'\t'+str(target)+'\t'+'0'+'\n')
        else:
            trust = Obs[e]
            source,target = e
            if trust == 1:
                trust_obs.write(str(source)+'\t'+str(target)+'\n')

    trust_obs.close()
    trust_targets.close()
    trust_truth.close()

    proc = subprocess.Popen(["./run.sh"])
    proc.communicate()
    proc.wait()



if __name__ == '__main__':
    pipeline()
    #test()
