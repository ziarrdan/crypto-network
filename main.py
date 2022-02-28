import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import os


def getData(start_date='2018-01-01', end_date='2022-01-01'):
    cryptos = pd.DataFrame()
    for filename in os.listdir("data-master/csv"):
        if 'csv' in filename:
            crypto = pd.read_csv("data-master/csv" + '/' + filename, encoding='Windows-1252')
            print(filename)
            crypto['Date'] = pd.to_datetime(crypto['Date'])
            crypto = crypto[(crypto['Date'] > start_date) & (crypto['Date'] < end_date)]
            cryptos[filename.split('.')[0]] = crypto['Price'].reset_index(drop=True)

        cryptos = cryptos.dropna(axis='columns')

    return cryptos


def calcEdges(cryptos, cutoff_corr=0.5):
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


def calcGraph(edges, nodes):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges)

    return G


cryptos = getData()
edges, nodes = calcEdges(cryptos)
graph = calcGraph(edges, nodes)