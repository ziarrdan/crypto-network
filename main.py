import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
import networkx as nx
import pandas as pd
import numpy as np
import os


class DataProvider():
    def get_data(self, start_date='2018-01-01', end_date='2022-01-01'):
        cryptos = pd.DataFrame()
        for filename in os.listdir("data-master/csv"):
            if 'csv' in filename:
                crypto = pd.read_csv("data-master/csv" + '/' + filename, encoding='Windows-1252')
                crypto['Date'] = pd.to_datetime(crypto['Date']).dt.date
                crypto = crypto[(crypto['Date'] >= start_date) & (crypto['Date'] <= end_date)].reset_index(drop=True)
                if not 'Date' in cryptos.columns:
                    cryptos['Date'] = crypto['Date']
                cryptos[filename.split('.')[0]] = crypto['Price'].reset_index(drop=True)

            cryptos = cryptos.dropna(axis='columns')
        cryptos = cryptos.set_index('Date', drop=True)

        return cryptos


    def get_data_for_dates(self, data, start_date='2018-01-01', end_date='2022-01-01'):
        data = data[(data.index > start_date) & (data.index < end_date)]

        return data


    def get_dates(self, start, end, increment):
        current = start
        list_dates = [start]

        while current + timedelta(days=increment) < end:
            list_dates.append(current + timedelta(days=10))
            current = current + timedelta(days=increment)

        return list_dates


class GraphProvider():
    def calc_edges(self, cryptos, cutoff_corr=0.5):
        cryptos_daily_returns = np.log(cryptos) - np.log(cryptos.shift(1))
        cryptos_correlations = cryptos_daily_returns.corr(method='pearson')
        cryptos_correlations = cryptos_correlations.mask(cryptos_correlations < cutoff_corr, 0)
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
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_weighted_edges_from(edges)

        return G


data_pro = DataProvider()
graph_pro = GraphProvider()
start_date = date(2019, 1, 1)
end_date = date(2021, 12, 30)
dates = data_pro.get_dates(start_date, end_date, 10)
cryptos = data_pro.get_data(start_date, end_date)

degree_centrality = []

for d in range(len(dates) - 1):
    start_int = dates[d]
    end_int = dates[d + 1]
    cryptos_int = data_pro.get_data_for_dates(cryptos, start_int, end_int)
    edges, nodes = graph_pro.calc_edges(cryptos_int)
    graph = graph_pro.calc_graph(edges, nodes)
    degree_centrality.append(nx.degree_centrality(graph))

plt.plot(dates[:-1], [d.get('Bitcoin') for d in degree_centrality])
plt.show()