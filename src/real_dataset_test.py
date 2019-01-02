import json
import matplotlib.pyplot as plt



comp_events={}
with open("../data/real_dataset/redteam.txt","r") as fp:
    for line in fp.readlines():
        terms=line.strip().split(",")
        comp_events[terms[0]]=(terms[2],terms[3])


count=0
edges={}
times={}
with open("/home/apdm02/data-sci-blow-2017/LUNA-2016/auth.txt","r") as fp:
    for line in fp:
        # if count>86400:
        #     break
        count+=1
        # print line
        terms=line.strip().split(",")
        # times.add(terms[0])
        # edges[(terms[3],terms[4])]=terms
        if int(terms[0])<86400:
            continue
        with open("/home/apdm02/workspace/git/data/cls_conflict/daily_auth/"+str(1+int(terms[0])/86400)+".txt","a+") as op:
            if comp_events.has_key(terms[0]) and terms[3]==comp_events[terms[0]][0] and terms[4]==comp_events[terms[0]][1]:
                op.write(line.strip()+","+"1\n")
            else:
                op.write(line.strip()+","+"0\n")


        # if times.has_key(terms[0]):
        #     times[terms[0]]+=1
        # else:
        #     times[terms[0]]=1
        if int(terms[0])>2557047+100:
            break


        if count%50000000==0:
            print count
#

print len(edges)
print count
# print len(times.keys()),min(times.values()),max(times.values())
# plt.hist(times.values(),bins=range(min(times.values()), max(times.values()) + 20, 20))
# plt.show()
# with open("edges.txt","w") as op:
#     for edge in edges.keys():
#         op.write(edge[0]+" "+edge[1]+"\n")


