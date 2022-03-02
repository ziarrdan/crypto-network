import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcol
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

    def calc_metric(self, func, dates, cryptos, data_pro, graph_pro):
        degree_centrality = []
        for d in range(len(dates) - 1):
            start_int = dates[d]
            end_int = dates[d + 1]
            cryptos_int = data_pro.get_data_for_dates(cryptos, start_int, end_int)
            edges, nodes = graph_pro.calc_edges(cryptos_int)
            graph = graph_pro.calc_graph(edges, nodes)
            degree_centrality.append(func(graph))
        degree_centrality_market = [sum(d.values()) / len(d) for d in degree_centrality]
        return degree_centrality, degree_centrality_market


class PlotProvider():
    def plot_metric(self, interes_list, dates, degree_metric, degree_metric_market, xlable, ylable, title):
        fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
        for coin, coin_color in interes_list.items():
            plt.plot(dates[:-1], [d.get(coin) for d in degree_metric], color=coin_color, linewidth=1.0,
                     linestyle='--',
                     label=coin)
        plt.plot(dates[:-1], degree_metric_market, color='k', linewidth=2.0, label='Market')
        myFmt = mdates.DateFormatter('%b %y')
        ax.xaxis.set_major_formatter(myFmt)
        plt.xticks(rotation=45)
        plt.legend(ncol=len(interes_list) + 1)
        plt.xlabel(xlable)
        plt.ylabel(ylable)
        plt.title(title, fontsize=12, y=1.03)
        plt.tight_layout()
        plt.show()

    def plot_network(self, cryptos, data_pro, degree_metric, graph_pro, start_date, end_date, title):
        cryptos_int = data_pro.get_data_for_dates(cryptos, start_date, end_date)
        edges, nodes = graph_pro.calc_edges(cryptos_int)
        graph = graph_pro.calc_graph(edges, nodes)
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        colors = [degree_metric[0][node] / max(degree_metric[0].values()) for node in graph.nodes()]
        sizes = [150 * i for i in colors]
        pos = nx.spring_layout(graph, k=2.5)
        cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName", ["blue", "green", "y", "orange"])
        sm = plt.cm.ScalarMappable(cmap=cm1)
        nx.draw_networkx_nodes(graph, pos, node_color=cm1(colors), node_size=sizes)
        nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color='k', width=0.25)
        plt.title(title, fontsize=18, y=1.03)
        fig.colorbar(sm, ax=None, orientation='vertical', shrink=0.75)
        plt.tight_layout()
        plt.show()

def run():
    data_pro = DataProvider()
    graph_pro = GraphProvider()
    plot_pro = PlotProvider()
    start_date = date(2019, 1, 1)
    end_date = date(2021, 8, 30)
    dates = data_pro.get_dates(start_date, end_date, 10)
    cryptos = data_pro.get_data(start_date, end_date)
    interes_list = {'Bitcoin': 'b', 'Ethereum': 'r', 'XPR': 'g', 'Cardano': 'orange'}

    degree_centrality, degree_centrality_market = graph_pro.calc_metric(
        nx.degree_centrality, dates, cryptos, data_pro, graph_pro)
    plot_pro.plot_metric(interes_list, dates, degree_centrality, degree_centrality_market,
                         xlable="Date", ylable="Degree Centrality", title="Degree Centrality")
    betweenness_centrality, betweenness_centrality_market = graph_pro.calc_metric(
        nx.betweenness_centrality, dates, cryptos, data_pro, graph_pro)
    plot_pro.plot_metric(interes_list, dates, betweenness_centrality, betweenness_centrality_market,
                         xlable="Date", ylable="Betweenness Centrality", title="Betweenness Centrality")

    plot_pro.plot_network(cryptos, data_pro, degree_centrality, graph_pro,
                          start_date=date(2020, 1, 1), end_date=date(2020, 1, 10), title='Degree centrality')
    plot_pro.plot_network(cryptos, data_pro, betweenness_centrality, graph_pro,
                          start_date=date(2020, 1, 1), end_date=date(2020, 1, 10), title='Betweenness centrality')


run()