import csv
import os
import random
import time
from os import listdir
import networkx as nx
from readAPDM import readAPDM


latlongNode = {}
tmcEdgetuple = {}
startTime = time.time()
vertices = []
edges = []
with open('TMC_Identification.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    nodeid = 0
    for row in reader:
        startPoint = (row['start_latitude'],row['start_longitude'])
        endPoint = (row['end_latitude'],row['end_longitude'])
        if startPoint not in latlongNode:
            latlongNode[startPoint] = nodeid
            vertices.append(nodeid)
            nodeid += 1
        if endPoint not in latlongNode:
            latlongNode[endPoint] = nodeid
            vertices.append(nodeid)
            nodeid += 1
        tmcEdgetuple[row['tmc']] = (latlongNode[startPoint], latlongNode[endPoint])
        print (latlongNode[startPoint],latlongNode[endPoint])
        edges.append((latlongNode[startPoint],latlongNode[endPoint]))

    G = nx.Graph()
    G.add_nodes_from(vertices)
    G.add_edges_from(edges)
    #print G.number_of_nodes()
    #print G.number_of_edges()
    #print nx.number_connected_components(G)
    graphs = list(nx.connected_component_subgraphs(G))

    CC = graphs[0]
    print 
    #print CC.number_of_nodes()
    #print CC.number_of_edges()
    #print sorted(nx.connected_components(G), key = len, reverse=True)[0]

print len(edges)    
count = 0
ed = []
for e in edges:
    source, target = e
    if source in CC.nodes() and target in CC.nodes():
        ed.append(e)

print len(set(ed))

#dataset = 'data/dc/'
#dataset = 'Processed-IRIX-Data-APDM/DC_2013-JunToDecByDay/'
dataset = 'Processed-IRIX-Data-APDM/DC_2014-JanFebByDay/'
#dataset = 'data/philly'
infiles = listdir(dataset)
C_edges = list(set(ed))

print len(C_edges)
for f in infiles:
    f_dir = dataset+f
    #print f_dir
    V,E,Obs,Omega = readAPDM(f_dir)
    count = 0
    #print Omega
    #print len(V)
    #for e in E:
    #    source,target = e
    #    e = (int(source),int(target))
    #    if e in ed:
    #        count += 1.0
    ecopy = [e for e in C_edges]  
    for e in ecopy:
        if e not in E:
            C_edges.remove(e)


print len(C_edges)
C = nx.Graph()
C.add_edges_from(C_edges)
print nx.is_connected(C)



'''
startTime = time.time()

# for in_dir in ['DC_2014-MarchByDay', 'DC_2014-JanFebByDay', 'DC_2013-JunToDecByDay']:
in_dir = 'DC_2014-MarchByDay'
filenames = listdir(in_dir)
out_dir = './Processed-IRIX-Data/' + in_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for filename in filenames:
    with open(in_dir+ '/'+filename) as f:
        next(f)
        tmcObslist = {}
        for line in f:
            line_list = line.split(',')
            speed = float(line_list[2])
            reference_speed = float(line_list[4])
            if speed < reference_speed:
                congestion = int(1)
            else:
                congestion = int(0)
            tmc = line_list[0]
            if tmc not in tmcObslist:
                tmcObslist[tmc] = []
            tmcObslist[tmc].append(congestion)

    # Vbig = range(0, len(latlongNode)) Ebig = tmcEdgetuple.values()  print "Ebig - E"  print len(list(set(Ebig)-set(E)))
    E = []
    Obs = {}
    Omega = {}
    edgeTupleList = []
    for tmc, edgeTuple in tmcEdgetuple.iteritems(): # since "TMC_identifications.csv" corresponds to DC_ALL_Month edge,
        if tmc in tmcObslist:  # the tmc in tmcEdgetuple is superset of tmc in tmcObslist(it corresponds to 'Reading-Oneday/Twoday...')
            if edgeTuple not in Obs:
                Obs[edgeTuple] = tmcObslist[tmc]
                Omega[edgeTuple] = [random.randrange(1,10+1),random.randrange(1,10+1)]
                E.append(edgeTuple)
    Vset = set()
    for e in E:
        Vset.add(e[0])
        Vset.add(e[1])
    V = list(Vset)
    Vbig = range(0, len(latlongNode))
    print len(Vbig) - len(V)
    # print 'E repeat edgeTuple number is '
    # print len(E) - len(list(set(E)))
    # print 'duplicated edges in E'
    # dupEdges = list( set([e for e in E if E.count(e) > 1]) )
    # for dupEdge in dupEdges:
    #     for tmc, edgeTuple in tmcEdgetuple.iteritems():
    #         if edgeTuple == dupEdge:
    #             print tmc, edgeTuple
    out_date = filename.split('.')[0][-10:]
    outfilepath = out_dir + "/APDM-IRIX-DC-"+out_date+".txt"
    writeAPDM(outfilepath, V, E, Obs, Omega)
    print out_date

print " takes %f seconds" % (time.time() - startTime)
'''
