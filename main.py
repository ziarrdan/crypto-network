import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcol
import matplotlib.cm as cm
from datetime import datetime, date, timedelta
from scipy import stats
import networkx as nx
import pandas as pd
import numpy as np
import os
import collections
import community as community_louvain


class DataProvider():
    def get_data(self, start_date='2018-01-01', end_date='2022-01-01'):
        cryptos = pd.DataFrame()
        for filename in os.listdir("data-master/csv"):
            if 'csv' in filename:
                crypto = pd.read_csv("data-master/csv" + '/' + filename, encoding='Windows-1252')
                if 'Date' in crypto.columns:
                    crypto['Date'] = pd.to_datetime(crypto['Date']).dt.date
                    crypto = crypto[(crypto['Date'] >= start_date) & (crypto['Date'] <= end_date)].reset_index(drop=True)
                    if not 'Date' in cryptos.columns:
                        cryptos['Date'] = crypto['Date']
                    cryptos[filename.split('.')[0]] = crypto['Open'].reset_index(drop=True)

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

    def calc_graphs(self, dates, cryptos, data_pro, graph_pro):
        graphs = []
        for d in range(len(dates) - 1):
            start_int = dates[d]
            end_int = dates[d + 1]
            cryptos_int = data_pro.get_data_for_dates(cryptos, start_int, end_int)
            edges, nodes = graph_pro.calc_edges(cryptos_int)
            graph = graph_pro.calc_graph(edges, nodes)
            graphs.append(graph)

        return graphs


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

    def plot_network(self, metric_func, graph, title):
        degree_metric = metric_func(graph)
        edges = graph.edges()
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        colors = [degree_metric[node] / max(degree_metric.values()) for node in graph.nodes()]
        sizes = [150 * i for i in colors]
        pos = nx.spring_layout(graph, k=2.5)
        cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName", ["blue", "green", "y", "orange"])
        sm = plt.cm.ScalarMappable(cmap=cm1)
        nx.draw_networkx_nodes(graph, pos, node_color=cm1(colors), node_size=sizes)
        nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color='k', width=0.05)
        plt.title(title, fontsize=18, y=1.03)
        fig.colorbar(sm, ax=None, orientation='vertical', shrink=0.75)
        plt.tight_layout()
        plt.show()

    def plot_degree_corr(self, g):
        avg_neighbor = nx.average_neighbor_degree(g)
        degree_sequence = dict(g.degree())
        degree_sequence_for_ks = {}
        for node in degree_sequence.keys():
            k = degree_sequence[node]
            if not k in degree_sequence_for_ks.keys():
                degree_sequence_for_ks[k] = [node]
            else:
                degree_sequence_for_ks[k].append(node)

        avg_neighbor_for_ks = {}
        for k in degree_sequence_for_ks.keys():
            nodes = degree_sequence_for_ks[k]
            sum_k = 0
            for n in nodes:
                sum_k += avg_neighbor[n]
            avg_neighbor_for_ks[k] = sum_k / len(nodes)
        avg_neighbor_for_ks = dict(sorted(avg_neighbor_for_ks.items()))

        fig3, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(avg_neighbor_for_ks.keys(), avg_neighbor_for_ks.values(), color="b", marker='.')
        plt.axhline(np.average(list(avg_neighbor_for_ks.values())), color="g", linestyle="--")
        plt.setp(ax, ylabel='$k_{nn}(k)$')
        plt.setp(ax, xlabel='k')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_yticks([10, 100, 1000])
        ax.set_yticklabels(['$10^1$', '$10^2$', '$10^3$'])
        ax.set_xticks([1, 10, 100, 1000])
        ax.set_xticklabels(['$10^0$', '$10^1$', '$10^2$', '$10^2$'])
        fig3.suptitle("Degree Correlation")
        fig3.tight_layout()
        plt.show()

        # k_nn_bar for a neutral degree is known to be (k_bar + (std / k_bar)), hence:
        k = list(dict(g.degree()).values())
        k_nn_bar = np.mean(k) + (np.std(k) / np.mean(k))

        dist_for_ttest = np.array(list(avg_neighbor_for_ks.values()))
        ttest_result = stats.ttest_1samp(dist_for_ttest, k_nn_bar)
        # print(ttest_result.pvalue)

    def plot_degree_dst(self, g):
        # Plot the out-degree distribution
        degree_sequence = reversed(sorted([d for n, d in g.degree()], reverse=True))  # degree sequence
        degreeCount = collections.Counter(degree_sequence)
        deg, in_cnt = zip(*degreeCount.items())
        cnt = np.array(in_cnt) / g.number_of_nodes()
        degree = [d for n, d in g.degree() if d != 0]

        fig1, axs = plt.subplots(ncols=2, nrows=2, figsize=(7, 7))
        axs[0, 0].scatter(deg, in_cnt, color="b", marker='.')
        axs[0, 0].title.set_text("Linear Scale")
        plt.setp(axs[0, 0], ylabel='pk')
        plt.setp(axs[0, 0], xlabel='k')
        plt.setp(axs[0, 0], ylim=([0, 0.1]))
        plt.setp(axs[0, 0], xlim=([0, 400]))
        axs[0, 0].set_yticks([0.1])
        axs[0, 0].set_yticklabels([0.1])
        axs[0, 0].set_xticks([10, 100, 200, 300, 400])
        axs[0, 0].set_xticklabels([10, 100, 200, 300, 400])

        axs[0, 1].scatter(deg, in_cnt, color="b", marker='.')
        axs[0, 1].title.set_text("Linear Binning")
        plt.setp(axs[0, 1], ylabel='pk')
        plt.setp(axs[0, 1], xlabel='k')
        axs[0, 1].set_yscale('log')
        axs[0, 1].set_xscale('log')
        axs[0, 1].set_yticks([0.001, 0.01, 0.1, 1])
        axs[0, 1].set_yticklabels(['$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$10^0$'])
        axs[0, 1].set_xticks([1, 10, 100, 1000])
        axs[0, 1].set_xticklabels(['$10^0$', '$10^1$', '$10^2$', '$10^3$'])

        n = 12
        b_n = 2 ** n
        bins = [2 ** n for n in range(n)]
        N = {}
        current_k = 1
        k_n = []
        for b in range(len(bins) - 1):
            k_n_cum = 0
            for i in range(bins[b], bins[b + 1]):
                if not b in N.keys():
                    N[b] = degreeCount[i]
                else:
                    N[b] += degreeCount[i]
                k_n_cum += i
            k_n.append(k_n_cum / (bins[b + 1] - bins[b]))

        values = [list(N.values())[i] / (bins[i] * g.number_of_nodes()) for i in range(len(N))]

        axs[1, 0].scatter(k_n, values, color="b", marker='.')
        axs[1, 0].scatter(deg, in_cnt, color="k", marker='.', alpha=0.05)
        axs[1, 0].title.set_text("Log-Binning")
        plt.setp(axs[1, 0], ylabel='pk')
        plt.setp(axs[1, 0], xlabel='k')
        axs[1, 0].set_yscale('log')
        axs[1, 0].set_xscale('log')
        axs[1, 0].set_yticks([0.0001, 0.001, 0.01, 0.1, 1])
        axs[1, 0].set_yticklabels(['$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$10^0$'])
        axs[1, 0].set_xticks([1, 10, 100, 1000])
        axs[1, 0].set_xticklabels(['$10^0$', '$10^1$', '$10^2$', '$10^2$'])

        cum_list = list(reversed(np.nancumsum(list(reversed(in_cnt)))))
        axs[1, 1].scatter(deg, cum_list, color="b", marker='.')
        axs[1, 1].title.set_text("Cumulative")
        plt.setp(axs[1, 1], ylabel='pk')
        plt.setp(axs[1, 1], xlabel='k')
        axs[1, 1].set_yscale('log')
        axs[1, 1].set_xscale('log')
        axs[1, 1].set_yticks([0.001, 0.01, 0.1, 1])
        axs[1, 1].set_yticklabels(['$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$10^0$'])
        axs[1, 1].set_xticks([1, 10, 100, 1000])
        axs[1, 1].set_xticklabels(['$10^0$', '$10^1$', '$10^2$', '$10^3$'])

        fig1.suptitle("In-Degree")
        fig1.tight_layout()
        plt.show()


    def plot_louvain(self, g, with_labels=False):
        commun_louvain = community_louvain.best_partition(g)
        pos = nx.spring_layout(g, k=1.5)
        cmap = cm.get_cmap('tab10', max(commun_louvain.values()) + 1)
        color_values = np.array(list(commun_louvain.values())) / max(commun_louvain.values())
        nx.draw_networkx_nodes(g, pos, commun_louvain.keys(), node_size=100,
                               cmap=cmap, node_color=list(color_values))
        nx.draw_networkx_edges(g, pos, alpha=0.5, width=0.05)
        if with_labels:
            labels = {}
            for node in g.nodes():
                labels[node] = node
            nx.draw_networkx_labels(g, pos, labels, font_size=9, font_color='w')
        plt.title("Community Detection on using Louvain method")
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

    graphs = graph_pro.calc_graphs(dates, cryptos, data_pro, graph_pro)

    degree_centrality = []
    for g in graphs:
        degree_centrality.append(nx.degree_centrality(g))
    degree_centrality_market = [sum(d.values()) / len(d) for d in degree_centrality]
    plot_pro.plot_metric(interes_list, dates, degree_centrality, degree_centrality_market,
                         xlable="Date", ylable="Degree Centrality", title="Degree Centrality")

    betweenness_centrality = []
    for g in graphs:
        betweenness_centrality.append(nx.betweenness_centrality(g))
    betweenness_centrality_market = [sum(d.values()) / len(d) for d in betweenness_centrality]
    plot_pro.plot_metric(interes_list, dates, betweenness_centrality, betweenness_centrality_market,
                         xlable="Date", ylable="Betweenness Centrality", title="Betweenness Centrality")

    plot_pro.plot_network(nx.degree_centrality, graphs[0], title='Degree centrality')
    plot_pro.plot_network(nx.betweenness_centrality, graphs[0], title='Betweenness centrality')
    plot_pro.plot_degree_corr(graphs[0])
    plot_pro.plot_degree_dst(graphs[0])
    plot_pro.plot_louvain(graphs[0])

run()