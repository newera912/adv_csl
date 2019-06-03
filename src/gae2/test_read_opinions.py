import pickle
import numpy as np
with open("opinion_points.pkl","r") as op:
    [belief,uncertain,test_mask]=pickle.load(op)

print("{} {} {}".format(len(belief),len(uncertain),len(test_mask)))

resutls={}
for i in range(len(belief)):
    if test_mask[i]==True:

        b=np.mean(belief[i])
        u=np.mean(uncertain[i])
        d=1-b-u
        resutls[i]=(b,d,u)
        print belief[i][0], 1 - belief[i][0] - uncertain[i][0], uncertain[i][0],b,d,u
with open("../../output/test/gcn_vae_opinions.pkl","wb") as fp:
    pickle.dump(resutls,fp)
