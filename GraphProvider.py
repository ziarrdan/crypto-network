import Graph
import networkx as nx
import numpy as np


class GraphProvider:
    def __init__(self, dates, cryptos, data_pro, window):
        self.dates = dates
        self.cryptos = cryptos
        self.data_pro = data_pro
        self.window = window

    def calc_edges(self, cryptos, cutoff_corr=0.5):
        cryptos_daily_returns = (np.log(cryptos) - np.log(cryptos.shift(1))) / np.std(cryptos)
        cryptos_correlations = cryptos_daily_returns.corr(method='pearson')
        cryptos_correlations = cryptos_correlations.mask(abs(cryptos_correlations) < cutoff_corr, 0)
        cryptos_correlations = cryptos_correlations.abs()
        nodes = set(cryptos_correlations.columns)
        edges = []
        for i, u in enumerate(cryptos_correlations):
            for j in range(i + 1, len(cryptos_correlations)):
                v = cryptos_correlations.columns[j]
                w = cryptos_correlations.iloc[i, j]
                if w != 0:
                    edges.append((u, v, cryptos_correlations.iloc[i, j]))

        return edges, nodes

    def calc_graph(self, edges, nodes):
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_weighted_edges_from(edges)

        return g

    def calc_graphs(self, cutoff_corr):
        graphs = []
        for d in range(self.window, len(self.dates)):
            start_int = self.dates[d - self.window]
            end_int = self.dates[d]
            cryptos_int = self.data_pro.get_data_for_dates(self.cryptos, start_int, end_int)
            edges, nodes = self.calc_edges(cryptos_int, cutoff_corr)
            temp_graph = Graph.Graph(self.calc_graph(edges, nodes), start_int, end_int)
            graphs.append(temp_graph)

        return graphs
