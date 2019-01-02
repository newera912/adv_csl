# import networkx as nx
import pickle,multiprocessing
import matplotlib.pyplot as plt
import time
import numpy as np
import json
import ternary
# from ternary import random_points
### Scatter Plot
scale = 40
figure, tax = ternary.figure(scale=scale)
tax.set_title("Scatter Plot", fontsize=20)
tax.boundary(linewidth=2.0)
tax.gridlines(multiple=5, color="blue")
# Plot a few different styles with a legend
points = random_points(30, scale=scale)
tax.scatter(points, marker='s', color='red', label="Red Squares")
points = random_points(30, scale=scale)
tax.scatter(points, marker='D', color='green', label="Green Diamonds")
tax.legend()
tax.ticks(axis='lbr', linewidth=1, multiple=5)

tax.show()


#
# pass
# T=10
# def read_running_time(file):
#     running_time_dict = {}
#     for weekday in range(1):
#         f = open(file, 'r')
#         for line in f:
#             try:
#                 result = json.loads(line)
#             except:
#                 print "time",line
#             key = result.keys()[0]
#             running_time = result[key]
#             if key not in running_time_dict:
#                 running_time_dict[key] = running_time
#     return running_time_dict
#
# ttime={}
#
# for ssize in [1000,5000,10000]:
#     time_dic=read_running_time("../output/running_time_"+str(ssize)+".json")
#     ttime[ssize]={}
#
#     for k,v in time_dic.items():  #k   1000_0.6_0.0_10_0_0.4_0.5_0
#         if int(k.split("_")[3])!=T :
#             # print "..."
#             continue
#         try:
#             ttr=float(k.split("_")[5])
#         except:
#             print k,v
#         if ttime[ssize].has_key(ttr):
#             ttime[ssize][ttr].append(float(v))
#         else:
#             ttime[ssize][ttr]=[float(v)]
#
#
# fig, ax = plt.subplots()
# for ssize in [1000,5000,10000]:
#     mses=[]
#     e_bar=[]
#     for ttr in [0.1,0.2,0.3,0.4,0.5]:
#         mses.append(map(int,(np.mean(ttime[ssize][ttr]),np.std(ttime[ssize][ttr]),np.min(ttime[ssize][ttr]),np.max(ttime[ssize][ttr]))))
#     print ssize,mses




# def chunks(llen, n):
#     # For item i in a range that is a length of l,
#     for i in range(0, llen, n):
#         # Create an index range for l of n items:
#         yield i,i+n
#
# R=range(10724535)
# m = int(np.ceil(float(len(R)/3.0) / 50))
# for (start_k, end_k)  in chunks(int(len(R)/3.0),m):
#     # (start_k, end_k) = (int(m * i), int(m * (i + 1)))
#     X=R[3*start_k:3*end_k]
#     print len(X),len(X)%3


# a=np.zeros((3,6))
# b=np.zeros((3,6))
# lambda1=0.5*np.ones((3,6))
#
# a[0]=0.11
# a[1]=0.22
# a[2]=0.33
# a[0][0]=0.9
# b[0]=0.4
# b[1]=0.5
# b[2]=0.6
# print a[0]*b[0]
# print "a-b",a-b,
# d=a-b
# d[d<0]=0.0
# d[d>0]=1.0
# print d
# a=a+lambda1*(b)
# print a
# print np.where(a>0.5,1.0,0.0)
#
# # [V, E, _] = pickle.load(open("/media/apdm05/data05/GasConsumption_Data_KDD2014_Zheng/proc_data/20130912ref_proc-0.6.pkl","rb"))
# [V, E] = pickle.load(open("../data/graph_data/nodes-47676.pkl","rb"))
#
# G= nx.Graph()
#
#
# G.add_edges_from(E)
# print len(G.nodes),nx.is_connected(G)
# comp=max(nx.connected_components(G), key=len)
#
# # for e in comp:
# #     print e
# print "#number conennected components:",len(comp),
# print "Top 10 component size",comp[:10]
# nx.draw(G)
# plt.show()
#
# seq = [1,2,3,4,5,6,7,8,9,10]
# pieces = 4
# m = float(len(seq))/pieces
# print [seq[int(m*i):int(m*(i+

