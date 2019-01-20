
import pickle
import numpy as np


"""
1 6876 154866
2 4112 47038
3 2034 18300
4 1737 18293
5 1555 15484
6 1278 10530
7 1200 10378
8 1043 8360
9 886 7856
10 675 4710
11 650 4276
300548
"""
def gen_OBS0():
    # for i in [1,2]:
    V=set()
    E=set()
    Obs={}
    T=5
    with open("epinions-11.txt","r") as op:
        for line in op:
            terms=line.strip().split()
            terms=map(int,terms)
            if terms[2]>5: continue
            V.add(terms[0])
            V.add(terms[1])
            e=(terms[0],terms[1])
            E.add(e)
            if Obs.has_key(e): print e,terms[2],"******************************"
            Obs[e] = [0.0] * T
            for i in range(terms[2] - 1, T):
                Obs[e][i] = 1.0

    print len(V), min(V), max(V)
    print len(E)
    pkl_file = open("epinions-11-"+str(T)+".pkl", 'wb')
    pickle.dump([list(V), list(E), Obs], pkl_file)
    pkl_file.close()

    print len(Obs),len(E)

def enron_analysisData():
    import networkx as nx
    G=nx.Graph()

    with open("email-Enron-graph-A.txt", "r") as op:
        for line in op:
            terms = line.strip().split()
            terms = map(int, terms)
            G.add_edge(terms[0],terms[1])
    print G.number_of_edges(),G.number_of_nodes()

    mcc_G=list(max(nx.connected_component_subgraphs(G), key=len))
    print mcc_G[0],mcc_G[-1],max(mcc_G),min(mcc_G)
    node2id={node:i for i,node in enumerate(mcc_G)}
    id2node={i:node for node,i in node2id.items()}

    V={}
    E={}
    for e in G.edges():
        if not node2id.has_key(e[0]) or not node2id.has_key(e[1]):
            continue
        ee1=(node2id[e[0]],node2id[e[1]])
        ee2=(node2id[e[1]], node2id[e[0]])
        E[ee1]=0
        E[ee2] = 0
        V[node2id[e[0]]]=0
        V[node2id[e[1]]]=0
    # 33696,180811
    print len(E),len(V),min(V.keys()),max(V.keys())
    N=len(V)
    for v in range(N):
        V[v+N]=1

    for (e1,e2) in E.keys():
        E[(e1+N,e2+N)]=1
    print len(E), len(V), min(V.keys()), max(V.keys())
    print V[33695],V[33696]
    with open("./enron/enron-graph.txt","w") as of:
        for (e1,e2) in sorted(E.keys()):
            of.write("{} {}\n".format(e1,e2))

def slashdot_analysisData():
    import networkx as nx
    G=nx.Graph()
    self_edges=0
    EE={}
    with open("Slashdot0902.txt", "r") as op:
        for line in op:
            if line.startswith("#"): continue
            terms = line.strip().split()
            terms = map(int, terms)
            if terms[0]==terms[1]:
                continue
            (e1,e2)=(terms[0],terms[1]) if terms[0]<terms[1] else (terms[1],terms[0])
            EE[(e1,e2)]=0
            G.add_edge(e1,e2)
    print G.number_of_edges(),G.number_of_nodes()
    print EE.keys()[:5],len(EE)
    mcc_G=list(max(nx.connected_component_subgraphs(G), key=len))
    print mcc_G[0],mcc_G[-1],max(mcc_G),min(mcc_G)
    node2id={node:i for i,node in enumerate(mcc_G)}
    id2node={i:node for node,i in node2id.items()}
    #504230
    V={}
    E={}
    for e in G.edges():
        if not node2id.has_key(e[0]) or not node2id.has_key(e[1]):
            continue
        ee1=(node2id[e[0]],node2id[e[1]])
        ee2=(node2id[e[1]], node2id[e[0]])
        E[ee1]=0
        E[ee2] = 0
        V[node2id[e[0]]]=0
        V[node2id[e[1]]]=0
    # 33696,180811
    print len(E),len(V),min(V.keys()),max(V.keys())
    N=len(V)
    for v in range(N):
        V[v+N]=1

    for (e1,e2) in E.keys():
        E[(e1+N,e2+N)]=1
    print len(E), len(V), min(V.keys()), max(V.keys())
    print V[82167],V[82168]
    with open("./slashdot/slashdot-graph.txt","w") as of:
        for (e1,e2) in sorted(E.keys()):
            of.write("{} {}\n".format(e1,e2))





def analysisData():
    V = set()
    E = set()
    time_edges={}
    time_nodes={}
    with open("epinions-11.txt", "r") as op:
        for line in op:
            terms = line.strip().split()
            terms = map(int, terms)
            V.add(terms[0])
            V.add(terms[1])
            e = (terms[0], terms[1])
            time_slot=terms[2]
            E.add(e)
            if time_nodes.has_key(time_slot):
                time_nodes[time_slot].add(terms[0])
                time_nodes[time_slot].add(terms[1])
                time_edges[time_slot].add(e)
            else:
                time_nodes[time_slot]=set()
                time_edges[time_slot]=set()
                time_nodes[time_slot].add(terms[0])
                time_nodes[time_slot].add(terms[1])
                time_edges[time_slot].add(e)

    for k, v in time_nodes.items():
        print k,len(v),len(time_edges[k])
    data=np.zeros((11,11))
    data_edges = np.zeros((11, 11))
    for k,v in time_nodes.items():
        for k1,v1 in time_nodes.items():
            data[k-1][k1-1]=len(v.intersection(v1))
            data_edges[k-1][k1-1]=len(time_edges[k].intersection(time_edges[k1]))
            # print k,len(v),len(time_edges[k])
    print data
    print data_edges
    print len(V),len(E)
def analysisData2():
    V = set()
    E = set()
    out_dgree={}
    in_dgree={}
    with open("epinions-11.txt", "r") as op:
        for line in op:
            terms = line.strip().split()
            terms = map(int, terms)
            if terms[2] != 1: continue
            V.add(terms[0])
            V.add(terms[1])
            e = (terms[0], terms[1])
            time_slot=terms[2]

            E.add(e)
            if out_dgree.has_key(terms[0]):
                out_dgree[terms[0]]+=1.0
            else:
                out_dgree[terms[0]]=1.0
            if in_dgree.has_key(terms[1]):
                in_dgree[terms[1]]+=1.0
            else:
                in_dgree[terms[1]]=1.0


            # print k,len(v),len(time_edges[k])
    for i,(k,v) in enumerate(sorted(out_dgree.iteritems(), key=lambda (k,v): (v,k))):
        print i,">>",k,v
    print len(out_dgree.keys()),len(in_dgree.keys()),len(list(set(out_dgree.keys()).union(set(in_dgree.keys())))),len(set(V).difference(list(set(out_dgree.keys()).union(set(in_dgree.keys())))))
    print len(V),len(E)
# analysisData()
# gen_OBS0()
enron_analysisData()
# slashdot_analysisData()
# analysisData2()