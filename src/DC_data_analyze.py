import csv
import os


root="/network/rit/lab/ceashpc/fenglab/baojian/git/data/DC_Data/"
datasets=["DC_2013-JunToDecByDay","DC_2014-JanFebByDay","DC_2014-MarchByDay"]

conf_score={}
cf_30=0
cf_100=0
cf_80=0
for dataset in datasets:
    for filename in os.listdir(root+dataset+"/"):
        if filename.endswith(".csv") :
            with open(root+dataset+"/"+filename) as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for r in reader:
                    if float(r[6]) !=30.0 and float(r[7])==100.0:
                        # print "ok"
                        cf_30+=1.0
                        # if float(r[7])!=100.0:
                        #     cf_100+=1
                        # if float(r[7])<80.0:
                        #     cf_80+=1
    print dataset,cf_30,cf_100,cf_80

print cf_30,cf_100,cf_80

print 100.0*(cf_100/cf_30)
print 100.0*(cf_80/cf_30)



                    # if not conf_score.has_key(r[6]):
                    #     conf_score[r[6]] = 1.0
                    # else:
                    #     conf_score[r[6]] += 1.0
#     print dataset
#
# for key, value in sorted(conf_score.iteritems(), key=lambda (k,v): (v,k)):
#     print "%s: %s" % (key, value)




