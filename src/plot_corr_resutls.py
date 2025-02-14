import json
import numpy as np
import ast
import matplotlib.pyplot as plt
import pickle
import ternary
import cPickle, bz2

#####testing one
def trans_corr1(trans_mat):
    corr_mat=np.zeros((len(trans_mat),len(trans_mat[0])))
    #([B],[B|B],[B|A])
    #rho_A->B = Mae[B|A] -Mae[B] / Mae[B|B] -Mae[B]
    for i in range(len(trans_mat)):
        for j in range(len(trans_mat[0])):
            # B_BA=np.abs(trans_mat[i][j][2]-trans_mat[i][j][0])
            # B_BB=np.abs(trans_mat[i][j][1]-trans_mat[i][j][0])
            B_BA = np.abs(trans_mat[i][j][2] - trans_mat[i][j][0])
            B_BB = np.abs(trans_mat[i][j][1] - trans_mat[i][j][0])
            # B_BA = trans_mat[i][j][2] - trans_mat[i][j][0]
            # B_BB = trans_mat[i][j][1] - trans_mat[i][j][0]

            if i==j  :
                corr_mat[i][j]=1.0
            elif np.abs(B_BB-B_BA)<0.01:#if the values did not change too much or very close
                corr_mat[i][j] =round(np.abs(B_BB-B_BA),3)
            else:
                corr_mat[i][j] = round(B_BA/(B_BB), 2)
            if corr_mat[i][j]>1.0:
                corr_mat[i][j]=1.0
            print i, j, trans_mat[i][j], B_BA, B_BB, corr_mat[i][j]

    return corr_mat

def trans_corr(trans_mat):
    corr_mat=np.zeros((len(trans_mat),len(trans_mat[0])))
    #([B],[B|B],[B|A])
    #rho_A->B = Mae[B|A] -Mae[B] / Mae[B|B] -Mae[B]
    for i in range(len(trans_mat)):
        for j in range(len(trans_mat[0])):
            B_BA=np.abs(trans_mat[i][j][2]-trans_mat[i][j][0])
            B_BB=np.abs(trans_mat[i][j][1]-trans_mat[i][j][0])
            if i==j:# or B_BB==B_BA :
                corr_mat[i][j]=1.0
            else:
                corr_mat[i][j] = round(B_BA /(B_BB+0.01), 2)
    for i in range(len(trans_mat)):
        for j in range(len(trans_mat[0])):
            if corr_mat[i][j]>1.0:
                # corr_mat[i][j]=round(np.abs(2.0-corr_mat[i][j]),2)
                corr_mat[i][j] = round(corr_mat[i][j]/np.max(corr_mat[i][j]), 2)
    print i, j, trans_mat[i][j],corr_mat[i][j]
    return corr_mat


def corr_heat_table_epinion_old():
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    # sphinx_gallery_thumbnail_number = 2

    source = ["None","Adv-COI", "CSL", "GCN-VAE"]
    target = ["None","Adv-COI", "CSL", "GCN-VAE"]
    dataset = "Epinions"
    dataset = "PA"
    dataset = "DC"
    """
    Source  Model: A    Target Model: B
    #([B],[B|B],[B|A])    
    """
    #epinions
    maes=[[(0.411,0.414,0.414),(0.206,0.206,0.206),(0.231,0.292,0.289),(0.267,0.26,0.26)],
          [(0.410,0.410,0.410),(0.206,0.206,0.206), (0.231,0.34,0.287), (0.26,0.26,0.26)],
          [(0.410,0.410,0.404),(0.206,0.206,0.208), (0.231,0.346,0.346), (0.268,0.432,0.432)],
          [(0.410,0.410,0.404),(0.206,0.294,0.282), (0.377,0.572,0.575), (0.268,0.432,0.432)]]

    #traffic Pa
    # maes = [[(,,),(0.079, 0.252, 0.252), (0.075, 0.413, 0.413), (0.121, 0.571, 0.440)],
    #                 [(,,),(0.079, 0.252, 0.254), (0.075, 0.413, 0.413), (0.121, 0.571, 0.398)],
    #                 [(,,),(0.079, 0.252, 0.267), (0.075, 0.413, 0.443), (0.121, 0.571, 0.571)],
    #       [(,,),(0.206,0.294,0.282), (0.377,0.572,0.575), (0.268,0.432,0.432)]]

    # traffic DC
    # maes = [[(,,),(0.052, 0.238, 0.238), (0.052, 0.238, 0.238), (0.133, 0.549, 0.491)],
    #         [(,,),(0.052, 0.238, 0.238), (0.052, 0.454, 0.454), (0.133, 0.549, 0.445)],
    #         [(,,),(0.052, 0.238, 0.244), (0.052, 0.454, 0.477), (0.133, 0.549, 0.549)],
    #         [(,,),(0.052, 0.238, 0.244), (0.052, 0.454, 0.477), (0.133, 0.549, 0.549)]]



    data_matrix = trans_corr(maes)

    fig, ax = plt.subplots()
    im = ax.imshow(data_matrix,cmap="YlGnBu")
    # Create colorbar
    cbarlabel = "Correlation"
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90,va="bottom",fontsize=14)
    # cbar.solids.set_edgecolor("RdBu_r")
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(source)))
    ax.set_yticks(np.arange(len(target)))
    ax.set_ylabel("Source Model")
    ax.set_xlabel("Target Model")
    # ... and label them with the respective list entries
    ax.set_xticklabels(target)
    ax.set_yticklabels(source,rotation=45)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    # for i in range(len(source)):
    #     for j in range(len(target)):
    #         text = ax.text(j, i, data_matrix[i, j],
    #                        ha="center", va="center", color="k")

    ax.set_title(dataset+" Correlation Between Models")
    fig.tight_layout()
    plt.show()
    fig.savefig("../output/plots/"+dataset+"_transfer_correlation-Jan13.png",dpi=500, bbox_inches = 'tight',pad_inches = 0.03)

def corr_heat_table_epinion():
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    # sphinx_gallery_thumbnail_number = 2

    source = ["Baseline0","Adv-COI", "CSL", "GCN-VAE"]
    target = ["Baseline0","Adv-COI", "CSL", "GCN-VAE"]
    dataset = ["Epinions","PA","DC"][0]
    """
    Source  Model: A    Target Model: B
    #([B],[B|B],[B|A])    
    #rho_A->B = Mae[B|A] -Mae[B] / Mae[B|B]-Mae[B]
    """

    results = {"Baseline0_Baseline0": [0.409, 0.409], "Baseline0_CSL": [0.23, 0.289], "Baseline0_GCN-VAE": [0.272, 0.271], "Baseline0_Adv-COI": [0.204, 0.206], \
               "CSL_Baseline0": [0.407, 0.407], "CSL_CSL": [0.232, 0.352], "CSL_GCN-VAE": [0.267, 0.276], "CSL_Adv-COI": [0.207, 0.207], \
               "GCN-VAE_Baseline0": [0.407, 0.407], "GCN-VAE_CSL": [0.232, 0.293], "GCN-VAE_GCN-VAE": [0.267, 0.287], "GCN-VAE_Adv-COI": [0.206, 0.206], \
               "Adv-COI_Baseline0": [0.407, 0.407], "Adv-COI_CSL": [0.232, 0.293], "Adv-COI_GCN-VAE": [0.267, 0.276], "Adv-COI_Adv-COI": [0.206, 0.207]}

    """new jan27 GCN-VAE update"""
    results={"Baseline0_Baseline0":[0.409, 0.409],"Baseline0_CSL":[0.23, 0.289],"Baseline0_GCN-VAE":[0.26, 0.29],"Baseline0_Adv-COI":[0.204, 0.212],\
            "CSL_Baseline0":[0.407, 0.407],"CSL_CSL":[0.232, 0.352],"CSL_GCN-VAE":[0.262, 0.279],"CSL_Adv-COI":[0.207, 0.229],\
            "GCN-VAE_Baseline0":[0.407, 0.407],"GCN-VAE_CSL":[0.232, 0.293],"GCN-VAE_GCN-VAE":[0.262, 0.429],"GCN-VAE_Adv-COI":[0.206, 0.229],\
            "Adv-COI_Baseline0":[0.407, 0.407],"Adv-COI_CSL":[0.232, 0.293],"Adv-COI_GCN-VAE":[0.262, 0.305],"Adv-COI_Adv-COI":[0.206, 0.228]}
    #epinions
    maes=[[[] for j in range(len(target))] for i in range(len(source)) ]
    for A in range(len(source)):
        for B in range(len(target)):
            val_B=results["{}_{}".format(target[B],target[B])][0]
            val_BB=results["{}_{}".format(target[B],target[B])][1]
            val_BA=results["{}_{}".format(source[A],target[B],)][1]
            maes[A][B]=(val_B,val_BB,val_BA)
    for row in maes:
        print row




    data_matrix = trans_corr(maes)

    fig, ax = plt.subplots()
    im = ax.imshow(data_matrix,cmap="YlGnBu") #PuOr  YlGnBu YlGn magma_r
    # Create colorbar
    cbarlabel = "Correlation"
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom",fontsize=14)
    # cbar.solids.set_edgecolor("RdBu_r")
    # We want to show all ticks...
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.set_xticks(np.arange(len(target)))
    ax.set_yticks(np.arange(len(source)))
    ax.set_ylabel("Source Model", fontsize=14)
    ax.set_xlabel("Target Model", fontsize=14)
    # ... and label them with the respective list entries
    ax.set_xticklabels(target)
    ax.set_yticklabels(source,rotation=45)
    # ax.tick_params(top=True, bottom=False,
    #                labeltop=True, labelbottom=False)
    for edge, spine in ax.spines.items():
        spine.set_visible(True)
    ax.set_xticks(np.arange(data_matrix.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data_matrix.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color='#afeeee', linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(source)):
        for j in range(len(target)):
            if data_matrix[i, j]<0.5:
                text = ax.text(j, i, data_matrix[i, j],
                               ha="center", va="center", color="k",fontdict={'fontsize' : 14, 'fontweight':'bold'})
            else:
                text = ax.text(j, i, data_matrix[i, j],
                               ha="center", va="center", color="w",fontdict={'fontsize' : 14, 'fontweight':'bold'})

    # ax.set_title(dataset+" Correlation Between Models")
    fig.tight_layout()
    plt.show()
    fig.savefig("../output/plots/"+dataset+"_transfer_correlation-May13-2.png",dpi=500, bbox_inches = 'tight',
    pad_inches = 0.03)

def corr_heat_table_traffic_pa():
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    # sphinx_gallery_thumbnail_number = 2

    source = ["Baseline0","Adv-COI", "CSL", "GCN-VAE"]
    target = ["Baseline0","Adv-COI", "CSL", "GCN-VAE"]
    dataset = ["Epinions","PA","DC"][1]
    """
    Source  Model: A    Target Model: B
    #([B],[B|B],[B|A])    
    #rho_A->B = Mae[B|A] -Mae[B] / Mae[B|B]-Mae[B]
    """
    results={"Baseline0_Baseline0":[0.404, 0.404],\
        "Baseline0_CSL":[0.075, 0.102],\
        "Baseline0_GCN-VAE":[0.14, 0.139],\
        "Baseline0_Adv-COI":[0.074, 0.085],\
        "CSL_Baseline0":[0.400, 0.400],\
        "CSL_CSL":[0.083, 0.179],\
        "CSL_GCN-VAE":[0.131, 0.256],\
        "CSL_Adv-COI":[0.079, 0.134],\
        "GCN-VAE_Baseline0":[0.400, 0.400],\
        "GCN-VAE_CSL":[0.083, 0.215],\
        "GCN-VAE_GCN-VAE":[0.131, 0.292],\
        "GCN-VAE_Adv-COI":[0.079, 0.143],\
        "Adv-COI_Baseline0":[0.400, 0.400],\
        "Adv-COI_CSL":[0.083, 0.123],\
        "Adv-COI_GCN-VAE":[0.135, 0.167],\
        "Adv-COI_Adv-COI":[0.079, 0.1]}
        #epinions
    maes=[[[] for j in range(len(target))] for i in range(len(source)) ]
    for A in range(len(source)):
        for B in range(len(target)):
            val_B=results["{}_{}".format(target[B],target[B])][0]
            val_BB=results["{}_{}".format(target[B],target[B])][1]
            val_BA=results["{}_{}".format(source[A],target[B],)][1]
            maes[A][B]=(val_B,val_BB,val_BA)
    for row in maes:
        print row




    data_matrix = trans_corr(maes)

    fig, ax = plt.subplots()
    im = ax.imshow(data_matrix, cmap="YlGnBu")  # PuOr  YlGnBu
    # Create colorbar
    cbarlabel = "Correlation"
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90,va="bottom",fontsize=14)
    # cbar.solids.set_edgecolor("RdBu_r")
    # We want to show all ticks...
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.set_xticks(np.arange(len(source)))
    ax.set_yticks(np.arange(len(target)))
    ax.set_ylabel("Source Model", fontsize=14)
    ax.set_xlabel("Target Model", fontsize=14)
    # ... and label them with the respective list entries
    ax.set_xticklabels(target)
    ax.set_yticklabels(source, rotation=45)
    # ax.tick_params(top=True, bottom=False,
    #                labeltop=True, labelbottom=False)
    for edge, spine in ax.spines.items():
        spine.set_visible(True)
    ax.set_xticks(np.arange(data_matrix.shape[1] + 1)- .5, minor=True)
    ax.set_yticks(np.arange(data_matrix.shape[0] + 1)- .5, minor=True)
    ax.grid(which="minor", color="#afeeee", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(source)):
        for j in range(len(target)):
            if data_matrix[i, j]<0.5:
                text = ax.text(j, i, data_matrix[i, j],
                               ha="center", va="center", color="k",fontdict={'fontsize' : 14, 'fontweight':'bold'})
            else:
                text = ax.text(j, i, data_matrix[i, j],
                               ha="center", va="center", color="w",fontdict={'fontsize' : 14, 'fontweight':'bold'})
    # ax.set_title(dataset+" Correlation Between Models")
    fig.tight_layout()
    plt.show()
    fig.savefig("../output/plots/PA_transfer_correlation-May13.png",dpi=500, bbox_inches = 'tight',pad_inches = 0.03)


def corr_heat_table_traffic_dc():
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    # sphinx_gallery_thumbnail_number = 2

    source = ["Baseline0","Adv-COI", "CSL", "GCN-VAE"]
    target = ["Baseline0","Adv-COI", "CSL", "GCN-VAE"]
    dataset = ["Epinions","PA","DC","Facebook","Enron","Slashdot"][2]
    """
    Source  Model: A    Target Model: B
    #([B],[B|B],[B|A])    
    #rho_A->B = Mae[B|A] -Mae[B] / Mae[B|B]-Mae[B]
    """
    results={"Baseline0_Baseline0":[0.431, 0.431],\
            "Baseline0_CSL":[0.056, 0.074],\
            "Baseline0_GCN-VAE":[0.122, 0.136],\
            "Baseline0_Adv-COI":[0.055, 0.058],\
            "CSL_Baseline0":[0.433, 0.433],\
            "CSL_CSL":[0.054, 0.146],\
            "CSL_GCN-VAE":[0.124, 0.209],\
            "CSL_Adv-COI":[0.052, 0.097],\
            "GCN-VAE_Baseline0":[0.433, 0.433],\
            "GCN-VAE_CSL":[0.054, 0.152],\
            "GCN-VAE_GCN-VAE":[0.124, 0.219],\
            "GCN-VAE_Adv-COI":[0.052, 0.099],\
            "Adv-COI_Baseline0":[0.433, 0.433],\
            "Adv-COI_CSL":[0.054, 0.13],\
            "Adv-COI_GCN-VAE":[0.124, 0.197],\
            "Adv-COI_Adv-COI":[0.052, 0.096]}
    #epinions
    maes=[[[] for j in range(len(target))] for i in range(len(source)) ]
    for A in range(len(source)):
        for B in range(len(target)):
            val_B=results["{}_{}".format(target[B],target[B])][0]
            val_BB=results["{}_{}".format(target[B],target[B])][1]
            val_BA=results["{}_{}".format(source[A],target[B],)][1]
            maes[A][B]=(val_B,val_BB,val_BA)
    for row in maes:
        print row




    data_matrix = trans_corr(maes)

    fig, ax = plt.subplots()
    im = ax.imshow(data_matrix, cmap="YlGnBu")  # PuOr  YlGnBu
    # Create colorbar
    cbarlabel = "Correlation"
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90,va="bottom",fontsize=14)
    # cbar.solids.set_edgecolor("RdBu_r")
    # We want to show all ticks...
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.set_xticks(np.arange(len(source)))
    ax.set_yticks(np.arange(len(target)))
    ax.set_ylabel("Source Model", fontsize=16)
    ax.set_xlabel("Target Model", fontsize=16)
    # ... and label them with the respective list entries
    ax.set_xticklabels(target)
    ax.set_yticklabels(source, rotation=45)
    # ax.tick_params(top=True, bottom=False,
    #                labeltop=True, labelbottom=False)
    for edge, spine in ax.spines.items():
        spine.set_visible(True)
    ax.set_xticks(np.arange(data_matrix.shape[1] + 1)- .5 , minor=True)
    ax.set_yticks(np.arange(data_matrix.shape[0] + 1)- .5 , minor=True)
    ax.grid(which="minor", color="#afeeee", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(source)):
        for j in range(len(target)):
            if data_matrix[i, j]<0.5:
                text = ax.text(j, i, data_matrix[i, j],
                               ha="center", va="center", color="k",fontdict={'fontsize' : 14, 'fontweight':'bold'})
            else:
                text = ax.text(j, i, data_matrix[i, j],
                               ha="center", va="center", color="w",fontdict={'fontsize' : 14, 'fontweight':'bold'})

    # ax.set_title(dataset+" Correlation Between Models")
    fig.tight_layout()
    plt.show()
    fig.savefig("../output/plots/DC_transfer_correlation-May13.png",dpi=500, bbox_inches = 'tight',pad_inches = 0.03)

def corr_heat_table_traffic_FB():
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    # sphinx_gallery_thumbnail_number = 2

    source = ["Baseline0","Adv-COI", "CSL", "GCN-VAE"]
    target = ["Baseline0","Adv-COI", "CSL", "GCN-VAE"]
    dataset = ["Epinions", "PA", "DC", "Facebook", "Enron", "Slashdot"][3]
    """
    Source  Model: A    Target Model: B
    #([B],[B|B],[B|A])    
    #rho_A->B = Mae[B|A] -Mae[B] / Mae[B|B]-Mae[B]
    """
    # results={"Baseline0_Baseline0":[0.401, 0.403],\
    #         "Baseline0_CSL":[0.118, 0.248],\
    #         "Baseline0_GCN-VAE":[0.115, 0.259],\
    #         "Baseline0_Adv-COI":[0.09, 0.192],\
    #         "CSL_Baseline0":[0.402, 0.402],\
    #         "CSL_CSL":[0.117, 0.351],\
    #         "CSL_GCN-VAE":[0.111, 0.396],\
    #         "CSL_Adv-COI":[0.09, 0.253],\
    #         "GCN-VAE_Baseline0":[0.402, 0.402],\
    #         "GCN-VAE_CSL":[0.117, 0.352],\
    #         "GCN-VAE_GCN-VAE":[0.111, 0.397],\
    #         "GCN-VAE_Adv-COI":[0.092, 0.26],\
    #         "Adv-COI_Baseline0":[0.401, 0.401],\
    #         "Adv-COI_CSL":[0.116, 0.349],\
    #         "Adv-COI_GCN-VAE":[0.111, 0.38],\
    #         "Adv-COI_Adv-COI":[0.093, 0.243]}

    results={"Baseline0_Baseline0":[0.394, 0.394],\
            "Baseline0_CSL":[0.118, 0.248],\
            "Baseline0_GCN-VAE":[0.115, 0.259],\
            "Baseline0_Adv-COI":[0.09, 0.192],\
            "CSL_Baseline0":[0.396, 0.396],\
            "CSL_CSL":[0.117, 0.351],\
            "CSL_GCN-VAE":[0.111, 0.396],\
            "CSL_Adv-COI":[0.09, 0.253],\
            "GCN-VAE_Baseline0":[0.396, 0.396],\
            "GCN-VAE_CSL":[0.117, 0.352],\
            "GCN-VAE_GCN-VAE":[0.111, 0.397],\
            "GCN-VAE_Adv-COI":[0.092, 0.26],\
            "Adv-COI_Baseline0":[0.397, 0.397],\
            "Adv-COI_CSL":[0.116, 0.349],\
            "Adv-COI_GCN-VAE":[0.111, 0.38],\
            "Adv-COI_Adv-COI":[0.093, 0.243]}
    #epinions
    maes=[[[] for j in range(len(target))] for i in range(len(source)) ]
    for A in range(len(source)):
        for B in range(len(target)):
            val_B=results["{}_{}".format(target[B],target[B])][0]
            val_BB=results["{}_{}".format(target[B],target[B])][1]
            val_BA=results["{}_{}".format(source[A],target[B],)][1]
            maes[A][B]=(val_B,val_BB,val_BA)
    for row in maes:
        print row




    data_matrix = trans_corr(maes)

    fig, ax = plt.subplots()
    im = ax.imshow(data_matrix, cmap="YlGnBu")  # PuOr  YlGnBu
    # Create colorbar
    cbarlabel = "Correlation"
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90,va="bottom",fontsize=14)
    # cbar.solids.set_edgecolor("RdBu_r")
    # We want to show all ticks...
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.set_xticks(np.arange(len(source)))
    ax.set_yticks(np.arange(len(target)))
    ax.set_ylabel("Source Model", fontsize=16)
    ax.set_xlabel("Target Model", fontsize=16)
    # ... and label them with the respective list entries
    ax.set_xticklabels(target)
    ax.set_yticklabels(source, rotation=45)
    # ax.tick_params(top=True, bottom=False,
    #                labeltop=True, labelbottom=False)
    for edge, spine in ax.spines.items():
        spine.set_visible(True)
    ax.set_xticks(np.arange(data_matrix.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data_matrix.shape[0] + 1)- .5, minor=True)
    ax.grid(which="minor", color="#afeeee", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(source)):
        for j in range(len(target)):
            if data_matrix[i, j]<0.5:
                text = ax.text(j, i, data_matrix[i, j],
                               ha="center", va="center", color="k",fontdict={'fontsize' : 14, 'fontweight':'bold'})
            else:
                text = ax.text(j, i, data_matrix[i, j],
                               ha="center", va="center", color="w",fontdict={'fontsize' : 14, 'fontweight':'bold'})
    # ax.set_title(dataset+" Correlation Between Models")
    fig.tight_layout()
    plt.show()
    fig.savefig("../output/plots/FB_transfer_correlation-May13.png",dpi=500, bbox_inches = 'tight',pad_inches = 0.03)

def corr_heat_table_traffic_Enron():
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    # sphinx_gallery_thumbnail_number = 2

    source = ["Baseline0","Adv-COI", "CSL", "GCN-VAE"]
    target = ["Baseline0","Adv-COI", "CSL", "GCN-VAE"]
    dataset = ["Epinions", "PA", "DC", "Facebook", "Enron", "Slashdot"][4]
    """
    Source  Model: A    Target Model: B
    #([B],[B|B],[B|A])    
    #rho_A->B = Mae[B|A] -Mae[B] / Mae[B|B]-Mae[B]
    """
    results={"Baseline0_Baseline0":[0.397, 0.397],\
    "Baseline0_CSL":[0.186, 0.231],\
    "Baseline0_GCN-VAE":[0.131, 0.274],\
    "Baseline0_Adv-COI":[0.12, 0.122],\
    "CSL_Baseline0":[0.396, 0.396],\
    "CSL_CSL":[0.189, 0.277],\
    "CSL_GCN-VAE":[0.132, 0.217],\
    "CSL_Adv-COI":[0.119, 0.145],\
    "GCN-VAE_Baseline0":[0.396, 0.396],\
    "GCN-VAE_CSL":[0.189, 0.352],\
    "GCN-VAE_GCN-VAE":[0.132, 0.396],\
    "GCN-VAE_Adv-COI":[0.119, 0.152],\
    "Adv-COI_Baseline0":[0.396, 0.396],\
    "Adv-COI_CSL":[0.181, 0.352],\
    "Adv-COI_GCN-VAE":[0.129, 0.395],\
    "Adv-COI_Adv-COI":[0.111, 0.145]}
    #epinions
    maes=[[[] for j in range(len(target))] for i in range(len(source)) ]
    for A in range(len(source)):
        for B in range(len(target)):
            val_B=results["{}_{}".format(target[B],target[B])][0]
            val_BB=results["{}_{}".format(target[B],target[B])][1]
            val_BA=results["{}_{}".format(source[A],target[B],)][1]
            maes[A][B]=(val_B,val_BB,val_BA)
    for row in maes:
        print row




    data_matrix = trans_corr(maes)

    fig, ax = plt.subplots()
    im = ax.imshow(data_matrix, cmap="YlGnBu")  # PuOr  YlGnBu
    # Create colorbar
    cbarlabel = "Correlation"
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90,va="bottom",fontsize=16)
    # cbar.solids.set_edgecolor("RdBu_r")
    # We want to show all ticks...
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.set_xticks(np.arange(len(source)))
    ax.set_yticks(np.arange(len(target)))
    ax.set_ylabel("Source Model", fontsize=16)
    ax.set_xlabel("Target Model", fontsize=16)
    # ... and label them with the respective list entries
    ax.set_xticklabels(target)
    ax.set_yticklabels(source, rotation=45)
    # ax.tick_params(top=True, bottom=False,
    #                labeltop=True, labelbottom=False)
    for edge, spine in ax.spines.items():
        spine.set_visible(True)
    ax.set_xticks(np.arange(data_matrix.shape[1] + 1) - .5 , minor=True)
    ax.set_yticks(np.arange(data_matrix.shape[0] + 1)- .5 , minor=True)
    ax.grid(which="minor", color="#afeeee", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(source)):
        for j in range(len(target)):
            if data_matrix[i, j]<0.5:
                text = ax.text(j, i, data_matrix[i, j],
                               ha="center", va="center", color="k",fontdict={'fontsize' : 14, 'fontweight':'bold'})
            else:
                text = ax.text(j, i, data_matrix[i, j],
                               ha="center", va="center", color="w",fontdict={'fontsize' : 14, 'fontweight':'bold'})

    # ax.set_title(dataset+" Correlation Between Models")
    fig.tight_layout()
    plt.show()
    fig.savefig("../output/plots/Enron_transfer_correlation-May13.png",dpi=500, bbox_inches = 'tight',pad_inches = 0.03)

def corr_heat_table_traffic_Slashdot():
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    # sphinx_gallery_thumbnail_number = 2

    source = ["Baseline0","Adv-COI", "CSL", "GCN-VAE"]
    target = ["Baseline0","Adv-COI", "CSL", "GCN-VAE"]
    dataset = ["Epinions", "PA", "DC", "Facebook", "Enron", "Slashdot"][5]
    """
    Source  Model: A    Target Model: B
    #([B],[B|B],[B|A])    
    #rho_A->B = Mae[B|A] -Mae[B] / Mae[B|B]-Mae[B]
    """
    results={"Baseline0_Baseline0":[0.409, 0.413],\
            "Baseline0_CSL":[0.188, 0.245],\
            "Baseline0_GCN-VAE":[0.146, 0.268],\
            "Baseline0_Adv-COI":[0.109, 0.139],\
            "CSL_Baseline0":[0.411, 0.411],\
            "CSL_CSL":[0.186, 0.297],\
            "CSL_GCN-VAE":[0.145, 0.234],\
            "CSL_Adv-COI":[0.106, 0.14],\
            "GCN-VAE_Baseline0":[0.412, 0.412],\
            "GCN-VAE_CSL":[0.186, 0.345],\
            "GCN-VAE_GCN-VAE":[0.145, 0.397],\
            "GCN-VAE_Adv-COI":[0.107, 0.15],\
            "Adv-COI_Baseline0":[0.412, 0.412],\
            "Adv-COI_CSL":[0.185, 0.349],\
            "Adv-COI_GCN-VAE":[0.147, 0.395],\
            "Adv-COI_Adv-COI":[0.107, 0.149]}
    #epinions
    maes=[[[] for j in range(len(target))] for i in range(len(source)) ]
    for A in range(len(source)):
        for B in range(len(target)):
            val_B=results["{}_{}".format(target[B],target[B])][0]
            val_BB=results["{}_{}".format(target[B],target[B])][1]
            val_BA=results["{}_{}".format(source[A],target[B],)][1]
            maes[A][B]=(val_B,val_BB,val_BA)
    for row in maes:
        print row




    data_matrix = trans_corr(maes)

    fig, ax = plt.subplots()
    im = ax.imshow(data_matrix, cmap="YlGnBu")  # PuOr  YlGnBu
    # Create colorbar
    cbarlabel = "Correlation"
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90,va="bottom",fontsize=14)
    # cbar.solids.set_edgecolor("RdBu_r")
    # We want to show all ticks...
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.set_xticks(np.arange(len(source)))
    ax.set_yticks(np.arange(len(target)))
    ax.set_ylabel("Source Model", fontsize=16)
    ax.set_xlabel("Target Model", fontsize=16)
    # ... and label them with the respective list entries
    ax.set_xticklabels(target)
    ax.set_yticklabels(source, rotation=45)
    # ax.tick_params(top=True, bottom=False,
    #                labeltop=True, labelbottom=False)
    for edge, spine in ax.spines.items():
        spine.set_visible(True)
    ax.set_xticks(np.arange(data_matrix.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data_matrix.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="#afeeee", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(source)):
        for j in range(len(target)):
            if data_matrix[i, j]<0.5:
                text = ax.text(j, i, data_matrix[i, j],
                               ha="center", va="center", color="k",fontdict={'fontsize' : 12, 'fontweight':'bold'})
            else:
                text = ax.text(j, i, data_matrix[i, j],
                               ha="center", va="center", color="w",fontdict={'fontsize' : 12, 'fontweight':'bold'})

    # ax.set_title(dataset+" Correlation Between Models")
    fig.tight_layout()
    plt.show()
    fig.savefig("../output/plots/Slashdot_transfer_correlation-May13.png",dpi=500, bbox_inches = 'tight',pad_inches = 0.03)

if __name__=='__main__':
    # corr_heat_table_epinion()
    # corr_heat_table_traffic_pa()
    #
    # corr_heat_table_traffic_dc()
    #
    # corr_heat_table_traffic_FB()

    corr_heat_table_traffic_Enron()

    # corr_heat_table_traffic_Slashdot()