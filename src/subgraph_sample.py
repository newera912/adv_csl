__author__ = 'Feng Chen'
import pickle
import matplotlib.pyplot as plt
import networkx as nx


"""
Using breadfirst search with the start vertex id 0, this function generates sampled networks of
size 500, 1000, 5000, 10000, and 47676, where 47676 is the size of the largest connected component
that includes vertex 0
"""
def sample_epinion_network(sample_sizes = [100,500, 1000, 5000, 10000, 47676]):
    filename = "data/trust-analysis/Epinions.txt"
    # print open(filename).readlines()[:4]
    dict_V = {}
    E = []
    for line in open(filename).readlines()[4:]:
        (str_start_v, str_end_v) = line.split()
        start_v = int(str_start_v)
        end_v = int(str_end_v)
        if not dict_V.has_key(start_v):
            dict_V[start_v] = 1
        if not dict_V.has_key(end_v):
            dict_V[end_v] = 1
        E.append((start_v, end_v))
    V = dict_V.keys()
    vertex_nns = {} #key is source node, value is a list of directed neighbors (targets)
    for v_start, v_end in E:
        if vertex_nns.has_key(v_start):
            if v_end not in vertex_nns[v_start]:
                vertex_nns[v_start].append(v_end)
        else:
            vertex_nns[v_start] = [v_end]
 
    sample_networks = []
    for sample_size in sample_sizes[:1]:
        sample_V, sample_E = breadth_first_search(vertex_nns, sample_size)
        print "sample network: #nodes: {0}, #edges: {1}".format(len(sample_V), len(sample_E))
        # print walk_V
        # print walk_E
        
        G = nx.DiGraph()
        G.add_nodes_from(sample_V)
        G.add_edges_from(sample_E)
        nx.draw(G)
        plt.show()
        '''
        sample_networks.append([sample_V, sample_E])
        pkl_file = open("data/trust-analysis/nodes-{0}.pkl".format(len(sample_V)), 'wb')
        pickle.dump([sample_V, sample_E], pkl_file)
        pkl_file.close()
        '''
 
"""
INPUT
vertex_nns: key is vertex id, and the value is a list of neighboring vertex ids
sample_size: the size of the sampled subgraph using breadth first search
 
We need to re-label the vertex ids of the sampled subgraph such that the ids are continuous starting from 0.
 
OUTPUT
V: a list of vertex ids
E: a list of pairs of vertex ids
"""
def breadth_first_search(vertex_nns, sample_size):
    sample_V = {}
    sample_E = []
    start_v = 0
    queue = [start_v]
    n = 0
    while len(queue) > 0:
        v = queue.pop()
        sample_V[v] = 1
        n = n + 1
        if vertex_nns.has_key(v):
            for v_n in vertex_nns[v]:
                if not sample_V.has_key(v_n) and v_n not in queue:
                    queue.append(v_n)
        if n >= sample_size:
            break

    for v, v_nns in vertex_nns.items():
        for v_n in v_nns:
            if sample_V.has_key(v) and sample_V.has_key(v_n):
                sample_E.append((v, v_n))
    old_id_2_new_id = {}
    for new_id, v in enumerate(sample_V.keys()):
        old_id_2_new_id[v] = new_id
    sample_v = range(len(sample_V))
    sample_e = [(old_id_2_new_id[v1], old_id_2_new_id[v2]) for (v1, v2) in sample_E]
    return sample_v, sample_e
 

def main():
    sample_epinion_network()
 

if __name__=='__main__':
    main()
