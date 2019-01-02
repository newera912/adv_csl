import networkx as nx
import json
import time

dataroot='/network/rit/lab/ceashpc/fenglab/baojian/git/subjectiveLogic'
class DataWrapper(object):
    def __init__(self, dataset, directed=True):
        if directed:
            if dataset == 'dc':
                graph_path = dataroot+'/DC_Data/dc_graph_directed.txt'
                self._data_folder = dataroot+'/DC_Data/hourly_data_directed/'
            elif dataset == 'philly':
                graph_path = dataroot+'/Philly_Data/philly_graph_directed.txt'
                self._data_folder = dataroot+'/Philly_Data/hourly_data_directed/'
            else:
                graph_path = ''
                print('dataset error.')
                exit(0)
        else:
            if dataset == 'dc':
                graph_path = dataroot+'/DC_Data/dc_graph_final.txt'
                self._data_folder = dataroot+'/DC_Data/hourly_data/'
            elif dataset == 'philly':
                graph_path = dataroot+'/Philly_Data/philly_graph_final.txt'
                self._data_folder = dataroot+'/Philly_Data/hourly_data/'
            else:
                graph_path = ''
                print('dataset error.')
                exit(0)
        self._graph_file_path = graph_path
        di_g = self._load_graph(dataset)
        nodes_list = list(di_g.nodes())
        nodes_map = dict()
        V = range(0, len(nodes_list))
        for i in V:
            nodes_map[nodes_list[i]] = i
        E = []
        edges_map = dict()
        for edge in di_g.edges(data=True):
            u_i = nodes_map[edge[0]]
            v_i = nodes_map[edge[1]]
            tmc_id = edge[2]['tmc_id']
            edge__ = (u_i, v_i)
            edges_map[tmc_id] = edge__
            E.append(edge__)
        self._edges_map = edges_map
        self._V = V
        self._E = E
        self._di_g = di_g
        # print('number of nodes: {}, number of edges: {}'.format(len(V), len(E)))

    def _data_wrapper(self, hour_, weekday_, precent=0.9):
        name_ = self._data_folder + './hour_{}_weekday_{}_speed.json'
        self._data_file_path = name_.format(hour_, weekday_)
        Obs = dict()
        #print('number of edges in edges_map: {}'.format(len(self._edges_map)))
        #print('number of edges : {}'.format(len(self._E)))
        with open(self._data_file_path) as f:
            for each_line in f.readlines():
                # print each_line
                items_ = json.loads(each_line)
                tmc_id = items_['tmc']
                indicator = []
                for item_ in items_['data']:
                    if item_:
                        if item_[0][0] < precent * item_[0][2]:
                            indicator.append(1)
                        else:
                            indicator.append(0)
                if len(indicator) >= 43:
                    indicator = indicator[:43]
                if len(indicator) < 43:
                    indicator = [0] * 43
                Obs[self._edges_map[tmc_id]] = indicator
        return self._V, self._E, Obs, self._di_g

    @property
    def g(self):
        return self._di_g

    def _load_graph(self, dataset):
        """
        :return: an directed graph.
        """
        di_g = nx.DiGraph()
        if dataset == 'dc':
            with open(self._graph_file_path) as f:
                all_lines = f.readlines()
                for each_line in all_lines[0:1383]:
                    items_ = json.loads(each_line)
                    (u, v) = items_['node']
                    di_g.add_node((u, v))
                for each_line in all_lines[1383:]:
                    items_ = json.loads(each_line)
                    node_i = (items_['u'][0], items_['u'][1])
                    node_j = (items_['v'][0], items_['v'][1])
                    id_ = items_['tmc']['tmc_id']
                    di_g.add_edge(node_i, node_j, tmc_id=id_)
        elif dataset == 'philly':
            with open(self._graph_file_path) as f:
                all_lines = f.readlines()
                for each_line in all_lines[0:603]:
                    items_ = json.loads(each_line)
                    (u, v) = items_['node']
                    di_g.add_node((u, v))
                for each_line in all_lines[603:]:
                    items_ = json.loads(each_line)
                    node_i = (items_['u'][0], items_['u'][1])
                    node_j = (items_['v'][0], items_['v'][1])
                    id_ = items_['tmc']['tmc_id']
                    di_g.add_edge(node_i, node_j, tmc_id=id_)
        else:
            print('dataset error.')
            exit(0)
        return di_g

    def get_data_case(self, hour_, weekday_, precent=0.9):
        """
        :param hour_: 5 <= hour_ <= 21
        :param weekday_: 0 <= weekday_ <= 4, i.e. ['Mon','Tue',...,'Fri']
        :param precent:
        :return:
        """
        return self._data_wrapper(hour_, weekday_, precent)

    @staticmethod
    def test_wrapper():
        # dataset='dc' or 'philly'
        dw = DataWrapper(dataset='dc', directed=True)
        V, E, Obs, G = dw.get_data_case(5, 2, 0.9)
        print(len(Obs))
        print(len(E))

        for edge in G.edges():
            if edge not in E:
                print(edge)
        for edge in E:
            if edge[0] == edge[1]:
                print('test')
        print('number of nodes: {}'.format(len(V)))
        print('number of edges: {}'.format(len(E)))
        print('number of cc: {}'.format(nx.is_weakly_connected(G)))


def main():
    DataWrapper.test_wrapper()


if __name__ == '__main__':
    main()
