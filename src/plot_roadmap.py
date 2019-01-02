import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap as Basemap

def read_data(filename):
    node2id={}
    id2loc={}
    edges={}
    with open(filename,"r") as op:
        for line in op.readlines():
            temp=json.loads(str(line))
            if temp.has_key("node: "):
                id2loc[temp['index']]=str("%s,%s"%(temp['node: '][0],temp['node: '][1]))
                node2id[str(("%s,%s"%(temp['node: '][0],temp['node: '][1])))]=temp['index']
            if temp.has_key("tmc"):
                u=str("%s,%s"%(temp['u'][0],temp['u'][1]))
                v = str("%s,%s" % (temp['v'][0], temp['v'][1]))
                uf = (float(temp['u'][0]), float(temp['u'][1]))
                vf = (float(temp['v'][0]), float(temp['v'][1]))
                edges[(node2id[u],node2id[v])]=(uf,vf)



    print len(node2id),node2id
    print len(id2loc),id2loc
    print len(edges),edges
    return edges

def matplot_plot():
    edges=read_data("../data/traffic_data/dc_graph_final.txt")

    lc = LineCollection(edges.values())

    fig, ax = plt.subplots()
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    ax.set_frame_on(False)

    plt.gca().add_collection(lc)
    plt.axis('equal')
    ax.grid(False)
    ax = plt.gca()

    fig.savefig('report2.png',dpi=600)
    # plt.show()




m = Basemap(
        projection='merc',
        llcrnrlon=-130,
        llcrnrlat=25,
        urcrnrlon=-60,
        urcrnrlat=50,
        lat_ts=0,
        resolution='i',
        suppress_ticks=True)

# position in decimal lat/lon
lats=[37.96,42.82]
lons=[-121.29,-73.95]
# convert lat and lon to map projection
mx,my=m(lons,lats)

# The NetworkX part
# put map projection coordinates in pos dictionary
G=nx.Graph()
G.add_edge('a','b')
pos={}
pos['a']=(mx[0],my[0])
pos['b']=(mx[1],my[1])
# draw
nx.draw_networkx(G,pos,node_size=200,node_color='blue')

# Now draw the map
m.drawcountries()
m.drawstates()
m.bluemarble()
plt.title('How to get from point a to point b')
plt.show()
plt.savefig('report3.png',dpi=600)

