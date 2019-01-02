import networkx as nx

def base1(V,E,sw_Obs,sw_Omega,E_X):
    Omega_X = {}
    for e in E_X:
        Omega_X[e] = (1.0,1.0)
    return Omega_X

def base2(V,E,sw_Obs,sw_Omega,E_X):
    Omega_X = {}
    for e in E_X:
        Omega_X[e] = (1.0,11.0)
    return Omega_X

def base3(V,E,sw_Obs,sw_Omega,E_X):
    Omega_X = {}
    for e in E_X:
        Omega_X[e] = (11.0,1.0)
    return Omega_X

