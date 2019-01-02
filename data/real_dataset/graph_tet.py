import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

# edges_list={}
# edges=set()
# with open("redteam.txt","r") as fp:
#     for line in fp.readlines():
#         terms=line.strip().split(",")
#         edges.add((terms[2],terms[3]))
#         if edges_list.has_key(terms[2]):
#             edges_list[terms[2]].append(terms[3])
#         else:
#             edges_list[terms[2]]=[terms[3]]
#
# print edges
# print edges_list
#
# G = nx.DiGraph()
# G.add_edges_from(list(edges))
# pos = nx.spring_layout(G)
#
# nx.draw(G,pos,arrowstyle='->')
# plt.show()

times={}
with open("redteam.txt","r") as fp:
    for line in fp.readlines():
        terms=line.strip().split(",")
        days=1+(int(terms[0])/86400)
        hours=1+(int(terms[0])%86400)/3600
        if times.has_key((days,hours)):
            times[(days,hours)]+=1
        else:
             times[(days,hours)]=1
        # if times.has_key(days):
        #     times[days]+=1
        # else:
        #      times[days]=1


for key, value in sorted(times.iteritems(), key=lambda (k,v): (v,k)):
    if key[0]==9:
        print "%s: %s" % (key, value)
#
edges={}
ed=set()
count=0
with open("/home/apdm02/workspace/git/data/cls_conflict/daily_auth/9.txt","r") as op:
    for line in op:
        terms=line.strip().split(",")
        if int(terms[0])>=766800 and int(terms[0])<=770400:
            edges[(terms[3],terms[4])]=1
            # ed.add((terms[3],terms[4]))
            if terms[-1]=="1":
                count+=1
print len(edges)
# print len(ed)
print count
