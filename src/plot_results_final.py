import json
import numpy as np
import ast
import matplotlib.pyplot as plt

def autolabel(ax,rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

def test_time():
    methods_label = ["csl", "CI-1", "CI-2", "sl", "base1", "base2", "base3"][:3]
    methods = ["csl", "csl-3-rules", "csl-3-rules-conflict-evidence", "sl", "base1", "base2", "base3"][:3]
    m_data={}
    time_graph = {}
    for method in methods:
        time_graph[method] = {}
        for s in [1000, 5000, 47676][:]:
            time_graph[method][s] = []

    for method in methods[:]:
        with open("../output/test/"+method+"_results-server-Apr30-47676.json") as fp:
            for line in fp:
                result=ast.literal_eval(line)
                graph_sizes=result['network_size']
                time_graph[method][graph_sizes].append(result['runtime'])

    for k,v in time_graph.items():
        for kk,vv in v.items():
            print k,kk,round(np.mean(vv),2),round(np.std(vv),2)


def test_new_bar_plot_traffic():
    methods_label = ["PSL","sl","csl", "CI-1", "CI-2", "base1", "base2", "base3"][:5]
    methods = ["psl","sl","csl", "csl-conflict-1", "csl-conflict-2",  "base1", "base2", "base3"][:5]

    # methods_label = ["CSL","CI","SL"][:]
    # methods = ["csl", "csl-3-rules-conflict-evidence", "sl"][:]

    m_data={}

    for method in methods[:]:
        with open("../output/test/traffic-2/"+method+"_results-server-traffic-T11-Sep03.json") as fp:
            m_data[method]=[]
            for line in fp:
                # print line
                if len(line.strip())<20: continue
                try:
                    result=ast.literal_eval(line)
                except:
                    print method
                    print line
                    return
                m_data[method].append(json.dumps(result))
    #{"b_mse": [0.19, 0.19], "prob_mse": [0.236, 0.0], "test_ratio": 0.5, "T": 8, "acc": [0.741, 0.0], "network_size": 1000, "alpha_mse": [1.89, 0.0], "positive_ratio": 0.05, "d_mse": [0.22139112903225805, 0.22139112903225805], "sample_size": 1, "realization": 2, "beta_mse": [1.8932258064516125, 0.0], "runtime": 9.420, "u_mse": [0.0, 0.0]}
    # print m_data["csl-3-rules-conflict-evidence"]
    #plot parameters
    colors=['b','c','y','g','r','k']
    N=7
    width=0.17
    ref_ratios = [0.8, 0.7, 0.6]
    datasets = ['philly', 'dc']


    idx=np.arange(N)
    for dataset in datasets[:]:
        for weekday in range(5)[:1]:
            for hour in range(6, 22)[2:3]:
                for ref_ratio in ref_ratios[:]:
                    for test_ratio in [0.1, 0.2, 0.3, 0.4, 0.5][:]:
                        # for method in methods[:]:
                        rects=[]
                        fig, ax = plt.subplots()
                        for i,method in enumerate(methods[:]):
                            prob_mse=[]
                            prob_std=[]
                            temp_prob_mse={}
                            for ratio_conflict in [0,0, 0.1, 0.2, 0.3, 0.4, 0.5,0.6][:]:
                                temp_prob_mse[ratio_conflict]=[]
                                for result in m_data[method]:
                                    # print result
                                    result=ast.literal_eval(result)
                                    if result['weekday']==weekday and result['hour']==hour and result['ref_ratio']==ref_ratio and result['test_ratio']==test_ratio and result['dataset']==dataset  and result['ratio_conflict']==ratio_conflict :
                                        temp_prob_mse[ratio_conflict].append(result['prob_mse'][0])
                            # print method,temp_prob_mse
                            for ratio_confl in sorted(temp_prob_mse.iterkeys()):
                                # print ratio_confl
                                prob_mse.append(np.mean(temp_prob_mse[ratio_confl]))
                                prob_std.append(np.std(temp_prob_mse[ratio_confl]))
                            print method,prob_mse
                            rects.append(ax.bar(idx + width*i, prob_mse, width, color=colors[i], yerr=prob_std))

                        ax.set_ylabel('Probability MAE')
                        ax.set_title('Comparison on Probability MAE ,'+dataset+'\n (ref_re='+str(ref_ratio)+' TestRatio='+str(test_ratio)+')')
                        ax.set_xticks(idx + width / 2)
                        ax.set_xticklabels(('0.0', '0.1', '0.2', '0.3', '0.4', '0.5','0.6'))
                        ax.set_xlabel('Percentage of edges with conflict evidence')
                        ax.set_yticks(np.arange(0, 0.61, 0.05))
                        ax.legend((rects), (methods_label),loc= 4)
                        #plt.show()
                        fig.savefig("../output/plots/Sep6-"+dataset+"-ref_ratio-"+str(ref_ratio)+"-test_ratio-"+str(test_ratio)+"-T11.png")
                        plt.close()
                        print "---- ref_Ratio:{} Test Ratio:{}".format(ref_ratio,test_ratio)

def timeslot_plot():
    # methods_label = ["PSL","SL","CSL", "CI-1", "CI-2",  "base1", "base2", "base3"][:5]
    # methods = ["psl","sl","csl", "csl-3-rules", "csl-3-rules-conflict-evidence", "base1", "base2", "base3"][:5]
    methods_label = ["GCN-VAE","PSL", "SL", "CSL", "CI"][:]
    methods = ["GCN-VAE","psl", "sl", "csl", "csl-3-rules-conflict-evidence"][:]
    # methods_label = ["CSL","CI","SL"][:]
    # methods = ["csl", "csl-3-rules-conflict-evidence", "sl"][:]

    m_data={}
    #1000 June5  5000 June7-5000 10000 June14-100000   /complete result-1
    for method in methods[1:]:
        with open("../output/test/complete result-4-gz/"+method+"_results-server-5000-0.2-tr.json") as fp:
            m_data[method]={}
            for line in fp:
                # print line
                if len(line.strip())<20: continue
                try:
                    result=ast.literal_eval(line)
                except:
                    print method
                    print line
                    return
                if m_data[method].has_key(result['T']):
                    if m_data[method][result['T']].has_key(result['positive_ratio']):
                        m_data[method][result['T']][result['positive_ratio']].append(json.dumps(result))
                    else:
                        m_data[method][result['T']][result['positive_ratio']]=[json.dumps(result)]
                else:
                    m_data[method][result['T']]={}
                    m_data[method][result['T']][result['positive_ratio']]=[json.dumps(result)]

    #{"b_mse": [0.19, 0.19], "prob_mse": [0.236, 0.0], "test_ratio": 0.5, "T": 8, "acc": [0.741, 0.0], "network_size": 1000, "alpha_mse": [1.89, 0.0], "positive_ratio": 0.05, "d_mse": [0.22139112903225805, 0.22139112903225805], "sample_size": 1, "realization": 2, "beta_mse": [1.8932258064516125, 0.0], "runtime": 9.420, "u_mse": [0.0, 0.0]}
    # print m_data["csl-3-rules-conflict-evidence"]
    #plot parameters

    colors=['g','b','c','k','r']
    marker=['s','o','X','d','*']
    colors = ['g','b', 'c', 'y', 'r']
    marker = ['d','^', 'h', 'X', '*']
    TT=[2,4,6,8, 9, 10, 11][3:]
    N=4
    width=0.15
    prob_mse0=[0.240,0.225,0.231,0.254]
    idx=TT
    for graph_sizes in [8518,1000, 5000,10000, 47676][2:3]:
        for test_ratio in [0.1, 0.2, 0.3, 0.4, 0.5][3:4]:
            for ratio in [0.2,0.6, 0.7, 0.8][:1]:
                for ratio_conflict in [0.0, 0.1, 0.2, 0.3, 0.4][2:3]:
                    rects=[]
                    fig, ax = plt.subplots()
                    ax.xaxis.set_tick_params(labelsize=12)
                    ax.yaxis.set_tick_params(labelsize=12)
                    print("GV ",prob_mse0)
                    ax.plot(TT, prob_mse0, color=colors[0], label=methods_label[0], marker=marker[0],
                            markeredgewidth=0.5, markeredgecolor='k')
                    for i,method in enumerate(methods[1:]):
                        i+=1
                        prob_mse=[]
                        prob_std=[]
                        temp_prob_mse={}
                        for T in TT[:]:
                            temp_prob_mse[T]=[]
                            # if not m_data[method].has_key(T) or not m_data[method][T].has_key(ratio): continue
                            for result in m_data[method][T][ratio]:
                                # print result
                                result=ast.literal_eval(result)
                                if result['network_size']==graph_sizes and result['T']==T and result['test_ratio']==test_ratio and result['positive_ratio']==ratio and result['ratio_conflict']==ratio_conflict:
                                    temp_prob_mse[T].append(result['prob_mse'][0])
                        # print method,temp_prob_mse
                        for T in sorted(temp_prob_mse.iterkeys()):
                            # print ratio_confl
                            prob_mse.append(np.mean(temp_prob_mse[T]))
                            prob_std.append(np.std(temp_prob_mse[T]))
                        print method,prob_mse,TT
                        # rects.append(ax.bar(idx + width*i, prob_mse, width, color=colors[i], yerr=prob_std))
                        ax.plot(TT,prob_mse,color=colors[i],label=methods_label[i],marker=marker[i],markeredgewidth=0.5, markeredgecolor='k') #

                    ax.set_ylabel('Probability MAE',fontsize=13)
                    # ax.set_title('Comparison on Probability MAE ,Graph-size='+str(graph_sizes)+'\n PosRatio='+str(ratio)+' Test Ratio='+str(test_ratio)+' ConflictRatio='+str(ratio_conflict)+')')
                    ax.set_xticks(idx)
                    ax.set_xticklabels(['50%','33%','25%','20%', '18%', '16%', '15%'][3:],fontsize=12)
                    ax.set_xlabel('Uncertainty Mass',fontsize=13)
                    ax.set_yticks(np.arange(0, 0.7, 0.1))
                    # y_ticks=[str(i) for i in np.arange(0, 0.7, 0.1)]
                    # ax.set_yticklabels(y_ticks, fontsize=12)
                    ax.legend((methods_label),loc= 2)
                    # leg.get_frame().set_alpha(0.5)
                    ax.grid(color='b', linestyle='-.', linewidth=0.05)
                    #plt.show()
                    fig.savefig("../output/plots/VaryingTW-5methods-TimeSlot-graphSize-"+str(graph_sizes)+"-PosRatio-"+str(ratio)+'-TestRatio-'+str(test_ratio)+"-ConflictRatio-"+str(ratio_conflict)+".png",dpi=360)
                    plt.close()

def TestRatio_plot():
    methods_label = ["PSL","SL","CSL", "CI-1", "CI-2",  "base1", "base2", "base3"][:5]
    methods = ["psl","sl","csl", "csl-3-rules", "csl-3-rules-conflict-evidence", "base1", "base2", "base3"][:5]
    methods_label = ["GCN-VAE", "PSL", "SL", "CSL", "CI"][:]
    methods = ["GCN-VAE", "psl", "sl", "csl", "csl-3-rules-conflict-evidence"][:]
    # methods_label = ["CSL","CI","SL"][:]
    # methods = ["csl", "csl-3-rules-conflict-evidence", "sl"][:]

    m_data={}
    #1000 June5  5000 June7-5000 10000 June14-100000   /complete result-1
    for method in methods[1:]:
        with open("../output/test/complete result-4-gz/"+method+"_results-server-5000-0.2-tr.json") as fp:
            m_data[method]={}
            for line in fp:
                # print line
                if len(line.strip())<20: continue
                try:
                    result=ast.literal_eval(line)
                except:
                    print method
                    print line
                    return
                if m_data[method].has_key(result['T']):
                    if m_data[method][result['T']].has_key(result['positive_ratio']):
                        m_data[method][result['T']][result['positive_ratio']].append(json.dumps(result))
                    else:
                        m_data[method][result['T']][result['positive_ratio']]=[json.dumps(result)]
                else:
                    m_data[method][result['T']]={}
                    m_data[method][result['T']][result['positive_ratio']]=[json.dumps(result)]

    #{"b_mse": [0.19, 0.19], "prob_mse": [0.236, 0.0], "test_ratio": 0.5, "T": 8, "acc": [0.741, 0.0], "network_size": 1000, "alpha_mse": [1.89, 0.0], "positive_ratio": 0.05, "d_mse": [0.22139112903225805, 0.22139112903225805], "sample_size": 1, "realization": 2, "beta_mse": [1.8932258064516125, 0.0], "runtime": 9.420, "u_mse": [0.0, 0.0]}
    # print m_data["csl-3-rules-conflict-evidence"]
    #plot parameters
    colors=['g','b','c','k','r']
    marker=['s','o','X','d','*']
    colors = ['g', 'b', 'c', 'y', 'r']
    marker = ['d', '^', 'h', 'X', '*']
    TR=[0.1, 0.2, 0.3, 0.4, 0.5]
    N=5
    width=0.15
    prob_mse0=[0.186,0.247,0.255,0.226,0.222]
    idx=TR
    for graph_sizes in [8518,1000, 5000,10000, 47676][2:3]:
        for T in [8, 9, 10, 11][1:2]:
            for ratio in [0.2,0.6, 0.7, 0.8][:1]:
                for ratio_conflict in [0, 0.1, 0.2, 0.3, 0.4][2:3]:
                    rects=[]
                    fig, ax = plt.subplots()
                    ax.xaxis.set_tick_params(labelsize=12)
                    ax.yaxis.set_tick_params(labelsize=12)
                    ax.plot(TR, prob_mse0, color=colors[0], label=methods_label[0], marker=marker[0],
                            markeredgewidth=0.5, markeredgecolor='k')
                    print("GV ", prob_mse0)
                    for i,method in enumerate(methods[1:]):
                        i+=1
                        prob_mse=[]
                        prob_std=[]
                        temp_prob_mse={}
                        for test_ratio in [0.1, 0.2, 0.3, 0.4, 0.5][:]:
                            temp_prob_mse[test_ratio]=[]
                            if not m_data[method].has_key(T) or not m_data[method][T].has_key(ratio): continue
                            for result in m_data[method][T][ratio]:
                                # print result
                                result=ast.literal_eval(result)
                                if result['network_size']==graph_sizes and result['T']==T and result['test_ratio']==test_ratio and result['positive_ratio']==ratio and result['ratio_conflict']==ratio_conflict:
                                    temp_prob_mse[test_ratio].append(result['prob_mse'][0])
                        # print method,temp_prob_mse
                        for test_ratio in sorted(temp_prob_mse.iterkeys()):
                            # print ratio_confl
                            prob_mse.append(np.mean(temp_prob_mse[test_ratio]))
                            prob_std.append(np.std(temp_prob_mse[test_ratio]))
                        print method,prob_mse,TR
                        # rects.append(ax.bar(idx + width*i, prob_mse, width, color=colors[i], yerr=prob_std))
                        ax.plot(TR,prob_mse,color=colors[i],label=methods_label[i],marker=marker[i],markeredgewidth=0.5, markeredgecolor='k')

                    ax.set_ylabel('Probability MAE',fontsize=13)
                    # ax.set_title('Comparison on Probability MAE ,Graph-size='+str(graph_sizes)+'\n PosRatio='+str(ratio)+' T ='+str(T)+' ConflictRatio='+str(ratio_conflict)+')')
                    ax.set_xticks(idx)
                    ax.set_xticklabels(['10%','20%','30%','40%','50%'],fontsize=12)
                    ax.set_xlabel('Test Ratio',fontsize=13)
                    ax.set_yticks(np.arange(0, 0.7, 0.1))
                    ax.legend((methods_label),loc= 2)
                    # leg.get_frame().set_alpha(0.5)
                    ax.grid(color='b', linestyle='-.', linewidth=0.05)
                    #plt.show()
                    fig.savefig("../output/plots/VaryingTestRatio-5methods-TestRatio-graphSize-"+str(graph_sizes)+"-PosRatio-"+str(ratio)+'-T'+str(T)+"-ConflictRatio-"+str(ratio_conflict)+".png",dpi=360)
                    plt.close()

def ConflictRatio_plot():
    methods_label = ["PSL","SL","CSL", "CI-1", "CI-2",  "base1", "base2", "base3"][:5]
    methods = ["psl","sl","csl", "csl-3-rules", "csl-3-rules-conflict-evidence", "base1", "base2", "base3"][:5]
    methods_label = ["GCN-VAE", "PSL", "SL", "CSL", "CI"][:]
    methods = ["GCN-VAE", "psl", "sl", "csl", "csl-3-rules-conflict-evidence"][:]
    # methods_label = ["CSL","CI","SL"][:]
    # methods = ["csl", "csl-3-rules-conflict-evidence", "sl"][:]

    m_data={}
    #1000 June5  5000 June7-5000 10000 June14-100000   /complete result-1
    for method in methods[1:]:
        with open("../output/test/complete result-4-gz/"+method+"_results-server-5000-0.2-tr.json") as fp:
            m_data[method]={}
            for line in fp:
                # print line
                if len(line.strip())<20: continue
                try:
                    result=ast.literal_eval(line)
                except:
                    print method
                    print line
                    return
                if m_data[method].has_key(result['T']):
                    if m_data[method][result['T']].has_key(result['positive_ratio']):
                        m_data[method][result['T']][result['positive_ratio']].append(json.dumps(result))
                    else:
                        m_data[method][result['T']][result['positive_ratio']]=[json.dumps(result)]
                else:
                    m_data[method][result['T']]={}
                    m_data[method][result['T']][result['positive_ratio']]=[json.dumps(result)]

    #{"b_mse": [0.19, 0.19], "prob_mse": [0.236, 0.0], "test_ratio": 0.5, "T": 8, "acc": [0.741, 0.0], "network_size": 1000, "alpha_mse": [1.89, 0.0], "positive_ratio": 0.05, "d_mse": [0.22139112903225805, 0.22139112903225805], "sample_size": 1, "realization": 2, "beta_mse": [1.8932258064516125, 0.0], "runtime": 9.420, "u_mse": [0.0, 0.0]}
    # print m_data["csl-3-rules-conflict-evidence"]
    #plot parameters
    colors=['g','b','c','k','r']
    marker=['s','o','X','d','*']
    colors = ['g', 'b', 'c', 'y', 'r']
    marker = ['d', '^', 'h', 'X', '*']
    CR=[0.0, 0.1, 0.2, 0.3, 0.4]
    N=5
    width=0.15
    prob_mse0=[0.175,0.236,0.248,0.260,0.326]
    idx=CR
    for graph_sizes in [8518,1000, 5000,10000, 47676][2:3]:
        for T in [8, 9, 10, 11][1:2]:
            for ratio in [0.2, 0.6, 0.7, 0.8][:1]:
                for test_ratio in [0.1, 0.2, 0.3, 0.4, 0.5][1:2]:
                    rects=[]
                    fig, ax = plt.subplots()
                    ax.xaxis.set_tick_params(labelsize=12)
                    ax.yaxis.set_tick_params(labelsize=12)
                    ax.plot(CR, prob_mse0, color=colors[0], label=methods_label[0], marker=marker[0],
                            markeredgewidth=0.5, markeredgecolor='k')
                    print("GV ", prob_mse0)
                    for i,method in enumerate(methods[1:]):
                        i+=1
                        prob_mse=[]
                        prob_std=[]
                        temp_prob_mse={}
                        for ratio_conflict in [0.0, 0.1, 0.2, 0.3, 0.4][:]:
                            temp_prob_mse[ratio_conflict]=[]
                            if not m_data[method].has_key(T) or not m_data[method][T].has_key(ratio): continue
                            for result in m_data[method][T][ratio]:
                                # print result
                                result=ast.literal_eval(result)
                                if result['network_size']==graph_sizes and result['T']==T and result['test_ratio']==test_ratio and result['positive_ratio']==ratio and result['ratio_conflict']==ratio_conflict:
                                    temp_prob_mse[ratio_conflict].append(result['prob_mse'][0])
                        # print method,temp_prob_mse
                        for ratio_conflict in sorted(temp_prob_mse.iterkeys()):
                            # print ratio_confl
                            prob_mse.append(np.mean(temp_prob_mse[ratio_conflict]))
                            prob_std.append(np.std(temp_prob_mse[ratio_conflict]))
                        print method,prob_mse,CR
                        # rects.append(ax.bar(idx + width*i, prob_mse, width, color=colors[i], yerr=prob_std))
                        ax.plot(CR,prob_mse,color=colors[i],label=methods_label[i],marker=marker[i],markeredgewidth=0.5, markeredgecolor='k')

                    ax.set_ylabel('Probability MAE',fontsize=13)
                    # ax.set_title('Comparison on Probability MAE ,Graph-size='+str(graph_sizes)+'\n PosRatio='+str(ratio)+' T ='+str(T)+' Test Ratio='+str(test_ratio))
                    ax.set_xticks(idx)
                    ax.set_xticklabels(['0%','10%','20%','30%','40%'],fontsize=12)
                    ax.set_xlabel('Conflict Ratio',fontsize=13)
                    ax.set_yticks(np.arange(0, 0.7, 0.1))

                    ax.legend((methods_label),loc= 2)
                    # leg.get_frame().set_alpha(0.5)
                    ax.grid(color='b', linestyle='-.', linewidth=0.05)
                    #plt.show()
                    fig.savefig("../output/plots/VaryingConflict-5methods-ConflictRatio-graphSize-"+str(graph_sizes)+"-PosRatio-"+str(ratio)+'-T'+str(T)+"-TestRatio-"+str(test_ratio)+".png",dpi=360)
                    plt.close()

def VaryGraphSize_plot():
    methods_label = ["PSL","SL","CSL", "CI-1", "CI-2",  "base1", "base2", "base3"][:5]
    methods = ["psl","sl","csl", "csl-3-rules", "csl-3-rules-conflict-evidence", "base1", "base2", "base3"][:5]
    methods_label = ["PSL", "SL", "CSL", "CI"][:]
    methods = ["psl", "sl", "csl", "csl-3-rules-conflict-evidence"][:]
    # methods_label = ["CSL","CI","SL"][:]
    # methods = ["csl", "csl-3-rules-conflict-evidence", "sl"][:]

    m_data={}
    #1000 June5  5000 June7-5000 10000 June14-100000   /complete result-1
    for method in methods[:]:
        m_data[method] = {}
        for gsize in [1000, 5000,10000, 47676][:]:
            with open("../output/test/complete result-4-gz/"+method+"_results-server-"+str(gsize)+"-0.2-tr.json") as fp:
                for line in fp:
                    # print line
                    if len(line.strip())<20: continue
                    try:
                        result=ast.literal_eval(line)
                    except:
                        print method
                        print line
                        return
                    if m_data[method].has_key(result['T']):
                        if m_data[method][result['T']].has_key(result['positive_ratio']):
                            m_data[method][result['T']][result['positive_ratio']].append(json.dumps(result))
                        else:
                            m_data[method][result['T']][result['positive_ratio']]=[json.dumps(result)]
                    else:
                        m_data[method][result['T']]={}
                        m_data[method][result['T']][result['positive_ratio']]=[json.dumps(result)]

    #{"b_mse": [0.19, 0.19], "prob_mse": [0.236, 0.0], "test_ratio": 0.5, "T": 8, "acc": [0.741, 0.0], "network_size": 1000, "alpha_mse": [1.89, 0.0], "positive_ratio": 0.05, "d_mse": [0.22139112903225805, 0.22139112903225805], "sample_size": 1, "realization": 2, "beta_mse": [1.8932258064516125, 0.0], "runtime": 9.420, "u_mse": [0.0, 0.0]}
    # print m_data["csl-3-rules-conflict-evidence"]
    #plot parameters
    colors=['g','b','c','k','r']
    marker=['s','o','X','d','*']
    GZ=[1000, 5000,10000, 47676]
    N=4
    width=0.15

    idx=np.arange(N)

    for T in [8, 9, 10, 11][2:3]:
        for ratio in [0.2,0.6, 0.7, 0.8][:1]:
            for test_ratio in [0.1, 0.2, 0.3, 0.4, 0.5][1:2]:
                for ratio_conflict in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6][2:3]:
                    rects=[]
                    fig, ax = plt.subplots()
                    for i,method in enumerate(methods[:]):
                        prob_mse=[]
                        prob_std=[]
                        temp_prob_mse={}
                        for graph_sizes in [1000, 5000, 10000, 47676]:
                            temp_prob_mse[graph_sizes]=[]
                            if not m_data[method].has_key(T) or not m_data[method][T].has_key(ratio): continue
                            for result in m_data[method][T][ratio]:
                                # print result
                                result=ast.literal_eval(result)
                                if result['network_size']==graph_sizes and result['T']==T and result['test_ratio']==test_ratio and result['positive_ratio']==ratio and result['ratio_conflict']==ratio_conflict:
                                    temp_prob_mse[graph_sizes].append(result['prob_mse'][0])
                        # print method,temp_prob_mse
                        for graph_sizes in sorted(temp_prob_mse.iterkeys()):
                            # print ratio_confl
                            prob_mse.append(np.mean(temp_prob_mse[graph_sizes]))
                            prob_std.append(np.std(temp_prob_mse[graph_sizes]))
                        print method,prob_mse,GZ
                        # rects.append(ax.bar(idx + width*i, prob_mse, width, color=colors[i], yerr=prob_std))
                        ax.plot(idx,prob_mse,color=colors[i],label=methods_label[i],marker=marker[i])

                    ax.set_ylabel('Probability MAE')
                    ax.set_title('Comparison on Probability MAE ,PosRatio='+str(ratio)+'\n T ='+str(T)+' Test Ratio='+str(test_ratio)+' ConflictRatio='+str(ratio_conflict))
                    ax.set_xticks(idx)
                    ax.set_xticklabels(['1000','5000','10000','47676'])
                    ax.set_xlabel('Graph Size')
                    ax.set_yticks(np.arange(0, 0.7, 0.1))
                    ax.legend((methods_label),loc= 2)
                    # leg.get_frame().set_alpha(0.5)
                    ax.grid(color='b', linestyle='-.', linewidth=0.2)
                    #plt.show()
                    fig.savefig("../output/plots/VaryingGraphSize-5methods-PosRatio-"+str(ratio)+'-T'+str(T)+"-TestRatio-"+str(test_ratio)+"-ConflictRatio-"+str(ratio_conflict)+".png")
                    plt.close()

def RealDataset_bar_plot():
    methods_label = ["PSL","SL","CSL", "CI-1", "CI-2"][:5]
    methods = ["PSL","SL","CSL", "CI-1", "CI-2"][:5]

    #traffic_tr:0.2,cr:0.3,0.6
    #epinions tr:0.2,cr:0.3 t:10
    results={"Philly":{"PSL":0.320,"SL":0.182,"CSL":0.124,"CI-1":0.099,"CI-2":0.101}, \
             "DC": {"PSL": 0.315, "SL": 0.181, "CSL": 0.117,"CI-1": 0.079, "CI-2": 0.084}, \
             "Epinions":{"PSL":0.292,"SL":0.328,"CSL":0.264,"CI-1":0.206,"CI-2":0.206},\
             "Facebook": {"PSL": 0.451, "SL": 0.200, "CSL": 0.152, "CI-1": 0.103, "CI-2": 0.094}, \
             "Enron": {"PSL": 0.257, "SL": 0.202, "CSL": 0.186, "CI-1": 0.120, "CI-2": 0.109}, \
             "SlashDot": {"PSL": 0.246, "SL": 0.189, "CSL": 0.176, "CI-1": 0.105, "CI-2": 0.103}}

    methods_label = ["GCN-VAE","PSL", "SL", "CSL", "CI"][:]
    methods = ["GCN-VAE","PSL", "SL", "CSL", "CI"][:]

    # traffic_tr:0.2,cr:0.3,0.6
    # epinions tr:0.2,cr:0.3 t:10
    results = {"Philly": {"GCN-VAE":0.114,"PSL": 0.320, "SL": 0.182, "CSL": 0.124,  "CI": 0.101}, \
               "DC": {"GCN-VAE":0.099,"PSL": 0.315, "SL": 0.181, "CSL": 0.117,  "CI": 0.084}, \
               "Epinions": {"GCN-VAE":0.266,"PSL": 0.292, "SL": 0.328, "CSL": 0.264,  "CI": 0.206}, \
               "Facebook": {"GCN-VAE":0.188,"PSL": 0.451, "SL": 0.292, "CSL": 0.152, "CI": 0.094}, \
               "Enron": {"GCN-VAE":0.188,"PSL": 0.257, "SL": 0.219, "CSL": 0.186, "CI": 0.109}, \
               "SlashDot": {"GCN-VAE":0.193,"PSL": 0.246, "SL": 0.197, "CSL": 0.176, "CI": 0.103}}
    datasets=["Philly","DC","Epinions","Facebook","Enron","SlashDot"]
    datasets_label = ["PA", "DC", "Epinions", "Facebook", "Enron", "SlashDot"]
    colors=['g','b','c','y','r','k']
    N=6
    width=0.15

    idx=np.arange(N)

    rects=[]
    fig, ax = plt.subplots()
    for i,method in enumerate(methods[:]):
        prob_mse=[]
        prob_std=[]
        temp_prob_mse={}
        for dataset in datasets:
            prob_mse.append(results[dataset][method])
        print method,prob_mse
        rects.append(ax.bar(idx + width*i, prob_mse, width, color=colors[i],linewidth=0.5,edgecolor = 'w'))

    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_ylabel('Probability MAE',fontsize=14)
    # ax.set_title('Real Dataset Comparison on Probability MAE')
    ax.set_xticks(idx + width / 2)
    ax.set_xticklabels(datasets_label, rotation=45,fontsize=14)
    # ax.set_xlabel('Dataset')
    ax.set_yticks(np.arange(0, 0.60, 0.05))
    leg=ax.legend((rects), (methods_label),loc= 2)
    leg.get_frame().set_alpha(0.8)
    ax.grid(color='b', linestyle='-.', linewidth=0.05)
    plt.show()
    fig.savefig("../output/plots/real_dataset-5methods-May25.png",dpi=360)
    # plt.close()


def plot_runtime_real_dataset():
    # results={"DC":{"PSL": 49.49, "SL": 1.89, "CSL": 3.63,"CI-1":9.01, "CI-2": 20.84}, \
    #          "Philly": {"PSL": 34.58, "SL": 0.85, "CSL": 1.22, "CI-1": 1.76, "CI-2": 3.97}, \
    #          "Epinions": {"PSL": 1503.0, "SL":65.56, "CSL": 15.11,"CI-1": 27.04, "CI-2": 82.37}, \
    #           "Facebook": {"PSL": 656.65, "SL": 9.05, "CSL": 146.39, "CI-1": 32.21, "CI-2": 258.36}, \
    #           "Enron": {"PSL": 3584.4, "SL": 69.10, "CSL": 362.29, "CI-1": 94.39, "CI-2": 658.02}, \
    #           "SlashDot": {"PSL": 16515.60, "SL":174.28 , "CSL": 963.103, "CI-1": 260.602, "CI-2": 1802.524}}
    # methods = ["PSL", "SL", "CSL", "CI-1", "CI-2"]
    results = {"Philly": {"GCN-VAE":1.35,"PSL": 34.58, "SL": 0.85, "CSL": 1.22, "CI": 3.97}, \
               "DC": {"GCN-VAE":2.28, "PSL": 49.49, "SL": 1.89, "CSL": 3.63, "CI": 20.84},
               "Epinions": {"GCN-VAE":797,"PSL": 1503.0, "SL": 65.56, "CSL": 15.11, "CI": 82.37}, \
               "Facebook": {"GCN-VAE":147,"PSL": 656.65, "SL": 930.98, "CSL": 146.39, "CI": 258.36}, \
               "Enron": {"GCN-VAE":1200.0,"PSL": 3584.4, "SL": 3315.17, "CSL": 362.29,  "CI": 658.02}, \
               "SlashDot": {"GCN-VAE":9221,"PSL": 16515.60, "SL": 11027.40, "CSL": 963.103,  "CI": 1802.524}}
    methods = ["GCN-VAE","PSL", "SL", "CSL", "CI"]
    datasets = [ "Philly", "DC","Epinions","Facebook","Enron","SlashDot"]
    datasets_label=[ "PA", "DC","Epinions","Facebook","Enron","SlashDot"]
    colors = ['g', 'b', 'c', 'y', 'r', 'k']
    N = 6
    width = 0.15

    idx = np.arange(N)

    rects = []
    fig, ax = plt.subplots()
    for i, method in enumerate(methods[:]):
        running_time = []
        for dataset in datasets:
            running_time.append(np.log(results[dataset][method]))
        print method, running_time,np.log(running_time)
        # rects.append(ax.bar(idx + width * i, running_time, width, color=colors[i],linewidth=0.5,edgecolor = 'w'))
        rects.append(ax.bar(idx + width * i, running_time, width, color=colors[i], linewidth=0.5, edgecolor='w'))


    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_ylabel('Running time (log scale in sec.)',fontsize=14)
    # ax.set_title('Running Time')
    ax.set_xticks(idx + width / 2)
    ax.set_xticklabels(datasets_label, rotation=45)
    # ax.set_yscale('log')
    # ax.set_xlabel('Dataset')
    # ax.set_yticks(np.arange(-1, 10))
    # ax.set_ylim([-1,10])
    leg = ax.legend((rects), (methods), loc=2)
    leg.get_frame().set_alpha(0.8)
    ax.grid(color='b', linestyle='-.', linewidth=0.05)
    plt.show()
    fig.savefig("../output/plots/real-world-running-time-log-5-methods-Dec18-min.png",dpi=360)
    # plt.close()

def plot_runtime_epinions():
    results={"1000":{"PSL": 71.0, "SL": 0.6, "CSL": 0.75,"CI-1": 1.87, "CI-2": 5.17}, \
             "5000": {"PSL": 623.0, "SL": 72.87, "CSL": 6.66, "CI-1": 12.69, "CI-2": 37.37}, \
             "10000": {"PSL": 1503.0, "SL":195.98, "CSL": 15.11,"CI-1": 27.04, "CI-2": 82.37}, \
             "47676": {"PSL": 0, "SL": 11562.42, "CSL": 5838.37,"CI-1": 1669.68, "CI-2": 4354.28}}
    methods = ["PSL", "SL", "CSL", "CI-1", "CI-2"]
    grapsizes = ["1000", "5000", "10000","47676"]
    colors = ['b', 'c', 'y', 'g', 'r', 'k']
    N = 4
    width = 0.07

    idx = np.arange(N)

    rects = []
    fig, ax = plt.subplots()
    for i, method in enumerate(methods[:]):
        running_time = []
        for graphsize in grapsizes:
            running_time.append(results[graphsize][method])
        print method, running_time,np.log(running_time)
        rects.append(ax.bar(idx + width * i, np.log(running_time), width, color=colors[i]))

    ax.set_ylabel('Log Running Time(seconds)')
    ax.set_title('Running Time')
    ax.set_xticks(idx + width / 2)
    ax.set_xticklabels(grapsizes)
    # ax.axes(yscale='log')
    ax.set_yticks(np.arange(-1, 10))
    ax.set_ylim([-1,10])
    leg = ax.legend((rects), (methods), loc=2)
    leg.get_frame().set_alpha(0.5)
    ax.grid(color='b', linestyle='-.', linewidth=0.2)
    # plt.show()
    fig.savefig("../output/plots/Epinions-running-time.png")
    plt.close()


def test_new_bar_plot():
    methods_label = ["PSL","SL","CSL", "CI-1", "CI-2",  "base1", "base2", "base3"][:5]
    methods = ["psl","sl","csl", "csl-3-rules", "csl-3-rules-conflict-evidence", "base1", "base2", "base3"][:5]

    # methods_label = ["CSL","CI","SL"][:]
    # methods = ["csl", "csl-3-rules-conflict-evidence", "sl"][:]

    m_data={}
    #1000 June5  5000 June7-5000 10000 June14-10000   /complete result-1   -Aug30-47676-0.2-opt
    for method in methods[:]:
        with open("../output/test/complete result-4/"+method+"_results-server-June14-10000.json") as fp:
            m_data[method]={}
            for line in fp:
                # print line
                if len(line.strip())<20: continue
                try:
                    result=ast.literal_eval(line)
                except:
                    print method
                    print line
                    return
                if m_data[method].has_key(result['T']):
                    if m_data[method][result['T']].has_key(result['positive_ratio']):
                        m_data[method][result['T']][result['positive_ratio']].append(json.dumps(result))
                    else:
                        m_data[method][result['T']][result['positive_ratio']]=[json.dumps(result)]
                else:
                    m_data[method][result['T']]={}
                    m_data[method][result['T']][result['positive_ratio']]=[json.dumps(result)]

    #{"b_mse": [0.19, 0.19], "prob_mse": [0.236, 0.0], "test_ratio": 0.5, "T": 8, "acc": [0.741, 0.0], "network_size": 1000, "alpha_mse": [1.89, 0.0], "positive_ratio": 0.05, "d_mse": [0.22139112903225805, 0.22139112903225805], "sample_size": 1, "realization": 2, "beta_mse": [1.8932258064516125, 0.0], "runtime": 9.420, "u_mse": [0.0, 0.0]}
    # print m_data["csl-3-rules-conflict-evidence"]
    #plot parameters
    colors=['b','c','y','g','r','k']
    N=7
    width=0.15

    idx=np.arange(N)
    for graph_sizes in [8518,1000, 5000,10000, 47676][2:3]:
        for T in [8,9,10,11][:]:
            for ratio in [0.6, 0.7, 0.8][:]:
                for test_ratio in [0.1, 0.2, 0.3, 0.4, 0.5][:]:
                    rects=[]
                    fig, ax = plt.subplots()
                    for i,method in enumerate(methods[:]):
                        prob_mse=[]
                        prob_std=[]
                        temp_prob_mse={}
                        for ratio_conflict in [0,0, 0.1, 0.2, 0.3, 0.4, 0.5,0.6][:]:
                            temp_prob_mse[ratio_conflict]=[]
                            if not m_data[method].has_key(T) or not m_data[method][T].has_key(ratio): continue
                            for result in m_data[method][T][ratio]:
                                # print result
                                result=ast.literal_eval(result)
                                if result['network_size']==graph_sizes and result['T']==T and result['test_ratio']==test_ratio and result['positive_ratio']==ratio and result['ratio_conflict']==ratio_conflict:
                                    temp_prob_mse[ratio_conflict].append(result['prob_mse'][0])
                        # print method,temp_prob_mse
                        for ratio_confl in sorted(temp_prob_mse.iterkeys()):
                            # print ratio_confl
                            prob_mse.append(np.mean(temp_prob_mse[ratio_confl]))
                            prob_std.append(np.std(temp_prob_mse[ratio_confl]))
                        print method,prob_mse
                        rects.append(ax.bar(idx + width*i, prob_mse, width, color=colors[i], yerr=prob_std))

                    ax.set_ylabel('Probability MAE')
                    ax.set_title('Comparison on Probability MAE ,Graph-size='+str(graph_sizes)+'\n (SnapShots='+str(T)+' TestRatio='+str(test_ratio)+' PosRatio='+str(ratio)+')')
                    ax.set_xticks(idx + width / 2)
                    ax.set_xticklabels(('0.0', '0.1', '0.2', '0.3', '0.4', '0.5','0.6'))
                    ax.set_xlabel('Percentage of edges with conflict evidence')
                    ax.set_yticks(np.arange(0, 0.61, 0.05))
                    leg=ax.legend((rects), (methods_label),loc= 4)
                    leg.get_frame().set_alpha(0.5)
                    ax.grid(color='b', linestyle='-.', linewidth=0.2)
                    #plt.show()
                    fig.savefig("../output/plots/Sep6-5methods-graphSize-"+str(graph_sizes)+"-SnapShots-"+str(T)+"-PosRatio-"+str(ratio)+"-test_ratio-"+str(test_ratio)+".png")
                    plt.close()

def test_new_bar_plot2():
    methods_label = ["PSL","SL","CSL", "CI-1", "CI-2",  "base1", "base2", "base3"][:5]
    methods = ["psl","sl","csl", "csl-3-rules", "csl-3-rules-conflict-evidence", "base1", "base2", "base3"][:5]

    # methods_label = ["CSL","CI","SL"][:]
    # methods = ["csl", "csl-3-rules-conflict-evidence", "sl"][:]

    m_data={}
    #1000 June5  5000 June7-5000 10000 June14-100000   /complete result-1
    for method in methods[:]:
        with open("../output/test/complete result-2/"+method+"_results-server-June7-5000.json") as fp:
            m_data[method]={}
            for line in fp:
                # print line
                if len(line.strip())<20: continue
                try:
                    result=ast.literal_eval(line)
                except:
                    print method
                    print line
                    return
                if m_data[method].has_key(result['T']):
                    if m_data[method][result['T']].has_key(result['positive_ratio']):
                        m_data[method][result['T']][result['positive_ratio']].append(json.dumps(result))
                    else:
                        m_data[method][result['T']][result['positive_ratio']]=[json.dumps(result)]
                else:
                    m_data[method][result['T']]={}
                    m_data[method][result['T']][result['positive_ratio']]=[json.dumps(result)]

    #{"b_mse": [0.19, 0.19], "prob_mse": [0.236, 0.0], "test_ratio": 0.5, "T": 8, "acc": [0.741, 0.0], "network_size": 1000, "alpha_mse": [1.89, 0.0], "positive_ratio": 0.05, "d_mse": [0.22139112903225805, 0.22139112903225805], "sample_size": 1, "realization": 2, "beta_mse": [1.8932258064516125, 0.0], "runtime": 9.420, "u_mse": [0.0, 0.0]}
    # print m_data["csl-3-rules-conflict-evidence"]
    #plot parameters
    colors=['b','c','y','g','r','k']
    N=5
    width=0.15

    idx=np.arange(N)
    for graph_sizes in [8518,1000, 5000,10000, 47676][2:3]:
        for T in [5,6,8,9,10,11][2:]:
            for ratio in [0.0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8][6:]:
                for ratio_conflict in [0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6][:1]:
                    rects=[]
                    fig, ax = plt.subplots()
                    for i,method in enumerate(methods[:]):
                        prob_mse=[]
                        prob_std=[]
                        temp_prob_mse={}
                        for test_ratio in [0.1, 0.2, 0.3, 0.4, 0.5][:]:
                            temp_prob_mse[test_ratio]=[]
                            if not m_data[method].has_key(T) or not m_data[method][T].has_key(ratio): continue
                            for result in m_data[method][T][ratio]:
                                # print result
                                result=ast.literal_eval(result)
                                if result['network_size']==graph_sizes and result['T']==T and result['test_ratio']==test_ratio and result['positive_ratio']==ratio and result['ratio_conflict']==ratio_conflict:
                                    temp_prob_mse[test_ratio].append(result['prob_mse'][0])
                        # print method,temp_prob_mse
                        for ratio_confl in sorted(temp_prob_mse.iterkeys()):
                            # print ratio_confl
                            prob_mse.append(np.mean(temp_prob_mse[ratio_confl]))
                            prob_std.append(np.std(temp_prob_mse[ratio_confl]))
                        print method,prob_mse
                        rects.append(ax.bar(idx + width*i, prob_mse, width, color=colors[i], yerr=prob_std))

                    ax.set_ylabel('Probability MAE')
                    ax.set_title('Comparison on Probability MAE ,Graph-size='+str(graph_sizes)+'\n (SnapShots='+str(T)+' ConflictRatio='+str(ratio_conflict)+' PosRatio='+str(ratio)+')')
                    ax.set_xticks(idx + width / 2)
                    ax.set_xticklabels(('0.1', '0.2', '0.3', '0.4', '0.5'))
                    ax.set_xlabel('Percentage of edges with conflict evidence')
                    ax.set_yticks(np.arange(0, 0.61, 0.05))
                    leg=ax.legend((rects), (methods_label),loc= 4)
                    leg.get_frame().set_alpha(0.5)
                    ax.grid(color='b', linestyle='-.', linewidth=0.2)
                    #plt.show()
                    fig.savefig("../output/plots/Aug28-5methods-graphSize-"+str(graph_sizes)+"-SnapShots-"+str(T)+"-PosRatio-"+str(ratio)+"-ConflictRatio="+str(ratio_conflict)+".png")
                    plt.close()



def test_testratio_bar_plot():
    methods_label = ["PSL","SL","CSL", "CI-1", "CI-2",  "base1", "base2", "base3"][:5]
    methods = ["psl","sl","csl", "csl-3-rules", "csl-3-rules-conflict-evidence", "base1", "base2", "base3"][:5]

    # methods_label = ["CSL","CI","SL"][:]
    # methods = ["csl", "csl-3-rules-conflict-evidence", "sl"][:]

    m_data={}
    #1000 June5  5000 June7-5000 10000 June14-100000   /complete result-1
    for method in methods[:]:
        with open("../output/test/complete result-1/"+method+"_results-server-June7-5000.json") as fp:
            m_data[method]={}
            for line in fp:
                # print line
                if len(line.strip())<20: continue
                try:
                    result=ast.literal_eval(line)
                except:
                    print method
                    print line
                    return
                if m_data[method].has_key(result['T']):
                    if m_data[method][result['T']].has_key(result['positive_ratio']):
                        m_data[method][result['T']][result['positive_ratio']].append(json.dumps(result))
                    else:
                        m_data[method][result['T']][result['positive_ratio']]=[json.dumps(result)]
                else:
                    m_data[method][result['T']]={}
                    m_data[method][result['T']][result['positive_ratio']]=[json.dumps(result)]

    #{"b_mse": [0.19, 0.19], "prob_mse": [0.236, 0.0], "test_ratio": 0.5, "T": 8, "acc": [0.741, 0.0], "network_size": 1000, "alpha_mse": [1.89, 0.0], "positive_ratio": 0.05, "d_mse": [0.22139112903225805, 0.22139112903225805], "sample_size": 1, "realization": 2, "beta_mse": [1.8932258064516125, 0.0], "runtime": 9.420, "u_mse": [0.0, 0.0]}
    # print m_data["csl-3-rules-conflict-evidence"]
    #plot parameters
    colors=['b','c','y','g','r','k']
    N=5
    width=0.15

    idx=np.arange(N)
    for graph_sizes in [8518,1000, 5000,10000, 47676][3:4]:
        for T in [5,6,8,9,10,11][2:]:
            for ratio in [0.0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8][6:]:
                for ratio_conflict in [0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6][:]:
                    rects=[]
                    fig, ax = plt.subplots()
                    for i,method in enumerate(methods[:]):
                        prob_mse=[]
                        prob_std=[]
                        temp_prob_mse={}
                        for test_ratio in [0.1, 0.2, 0.3, 0.4, 0.5][:]:
                            temp_prob_mse[test_ratio]=[]
                            if not m_data[method].has_key(T) or not m_data[method][T].has_key(ratio): continue
                            for result in m_data[method][T][ratio]:
                                # print result
                                result=ast.literal_eval(result)
                                if result['network_size']==graph_sizes and result['T']==T and result['test_ratio']==test_ratio and result['positive_ratio']==ratio and result['ratio_conflict']==ratio_conflict:
                                    temp_prob_mse[test_ratio].append(result['prob_mse'][0])
                        # print method,temp_prob_mse
                        for ratio_confl in sorted(temp_prob_mse.iterkeys()):
                            # print ratio_confl
                            prob_mse.append(np.mean(temp_prob_mse[ratio_confl]))
                            prob_std.append(np.std(temp_prob_mse[ratio_confl]))
                        print method,prob_mse
                        rects.append(ax.bar(idx + width*i, prob_mse, width, color=colors[i], yerr=prob_std))

                    ax.set_ylabel('Probability MAE')
                    ax.set_title('Comparison on Probability MAE ,Graph-size='+str(graph_sizes)+'\n (SnapShots='+str(T)+' Conflict Ratio='+str(ratio_conflict)+' PosRatio='+str(ratio)+')')
                    ax.set_xticks(idx + width / 2)
                    ax.set_xticklabels(('0.1', '0.2', '0.3', '0.4', '0.5'))
                    ax.set_xlabel('Percentage of edges with conflict evidence')
                    ax.set_yticks(np.arange(0, 0.61, 0.05))
                    leg=ax.legend((rects), (methods_label),loc= 4)
                    leg.get_frame().set_alpha(0.5)
                    ax.grid(color='b', linestyle='-.', linewidth=0.2)
                    #plt.show()
                    fig.savefig("../output/plots/Aug22-5methods-graphSize-"+str(graph_sizes)+"-SnapShots-"+str(T)+"-PosRatio-"+str(ratio)+"-Conflict Ratio-"+str(ratio_conflict)+".png")
                    plt.close()

if __name__=='__main__':


    # plot_runtime_epinions()

    # timeslot_plot()
    # TestRatio_plot()
    # ConflictRatio_plot()

    # VaryGraphSize_plot()
    RealDataset_bar_plot()
    # plot_runtime_real_dataset()