from datetime import datetime, timedelta, date
import community as community_louvain
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcol
import matplotlib.cm as cm
from scipy import stats
import collections
import community
import networkx as nx
import pandas as pd
import numpy as np
import os


class Graph:
    def __init__(self, graph, start, end):
        self.graph = graph
        self.start = start
        self.end = end

    def get_graph(self):
        return self.graph

    def get_dates(self):
        return self.start, self.end


class DataProvider():
    def get_data(self, start_date='2018-01-01', end_date='2022-01-01'):
        cryptos = pd.DataFrame()
        for filename in os.listdir("data-master/csv"):
            if 'csv' in filename:
                crypto = pd.read_csv("data-master/csv" + '/' + filename, encoding='Windows-1252')
                if 'Date' in crypto.columns:
                    crypto['Date'] = pd.to_datetime(crypto['Date']).dt.date
                    crypto = crypto[(crypto['Date'] >= start_date) & (crypto['Date'] <= end_date)].reset_index(drop=True)
                    if len(crypto['Date']) > 0 and crypto['Date'].iloc[0] == start_date:
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
            list_dates.append(current + timedelta(days=increment))
            current = current + timedelta(days=increment)

        return list_dates


class GraphProvider():
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
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_weighted_edges_from(edges)

        return G

    def calc_graphs(self, dates, cryptos, data_pro, graph_pro, window):
        graphs = []
        for d in range(window, len(dates)):
            start_int = dates[d - window]
            end_int = dates[d]
            cryptos_int = data_pro.get_data_for_dates(cryptos, start_int, end_int)
            edges, nodes = graph_pro.calc_edges(cryptos_int)
            graph = Graph(graph_pro.calc_graph(edges, nodes), start_int, end_int)
            graphs.append(graph)

        return graphs


class PlotProvider():
    def loseAxes(self, ax, group=1):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if group == 2:
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        else:
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

        return ax


    def plot_metric_bull(self, dates, degree_metric, xlable, ylable, title, window, cycles):
        fig, ax = plt.subplots(figsize=(8, 4), dpi=600)
        plt.plot(dates[window:], np.array(degree_metric), color='k', linewidth=1.25)
        for start, end in cycles:
            plt.axvline(x=start, color='g', linestyle='--', linewidth=1)
            plt.axvline(x=end, color='r', linestyle='--', linewidth=1)
        myFmt = mdates.DateFormatter('%b %y')
        ax = self.loseAxes(ax, 1)
        ax.xaxis.set_major_formatter(myFmt)
        plt.yticks(fontsize=14)
        plt.xticks(rotation=25, fontsize=14)
        plt.xlabel(xlable, fontsize=14)
        plt.ylabel(ylable, fontsize=14)
        plt.title(title, fontsize=14, y=1.03)
        plt.tight_layout()
        plt.savefig('pics/' + title + '.png')
        plt.show()
        plt.close()

    def plot_metric(self, dates, degree_metric, degree_metric_market, xlable, ylable, title, window, interes_list={}, major_events={}):
        fig, ax = plt.subplots(figsize=(8, 4), dpi=600)
        for coin, coin_color in interes_list.items():
            plt.plot(dates[window:], [d.get(coin) for d in degree_metric], color=coin_color, linewidth=.5,
                     linestyle='--',
                     label=coin)
        plt.plot(dates[window:], degree_metric_market, color='k', linewidth=1.25, label='Market')
        for key in major_events.keys():
            plt.axvline(x=key, color='k', linestyle='--', linewidth=1)
        if len(interes_list) == 0:
            ax.set_xticks(list(major_events.keys()))
        myFmt = mdates.DateFormatter('%b %y')
        ax = self.loseAxes(ax, 1)
        ax.xaxis.set_major_formatter(myFmt)
        plt.yticks(fontsize=14)
        plt.xticks(rotation=25, fontsize=14)
        plt.legend(ncol=len(interes_list) + 1)
        plt.xlabel(xlable, fontsize=14)
        plt.ylabel(ylable, fontsize=14)
        plt.title(title, fontsize=14, y=1.03)
        plt.tight_layout()
        plt.savefig('pics/' + title + '.png')
        plt.show()
        plt.close()


    def plot_network(self, metric_func, graph, plot_network, title, show_labels=False):
        degree_metric = metric_func(graph)
        fig, ax = plt.subplots(figsize=(8, 7), dpi=600)
        ax = self.loseAxes(ax, 2)
        edges = graph.edges()
        colors = [degree_metric[node] / max(degree_metric.values()) for node in graph.nodes()]
        if show_labels:
            sizes = [500 for _ in colors]
        else:
            sizes = [200 * i if i > 0 else 10 for i in colors]
        cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName", ["navy", "green", "yellow", "red"])
        sm = plt.cm.ScalarMappable(cmap=cm1)
        nx.draw_networkx_nodes(nx.MultiDiGraph(graph), pos=plot_network, node_color=cm1(colors), node_size=sizes)
        nx.draw_networkx_edges(nx.MultiDiGraph(graph), pos=plot_network, edgelist=edges, edge_color='k', width=0.5, connectionstyle="arc3,rad=0.25", arrowstyle='-', alpha=0.25)
        if show_labels:
            labels = {}
            for node in graph.nodes():
                labels[node] = node
            nx.draw_networkx_labels(graph, plot_network, labels, font_size=8, font_color='white')
        plt.title(title, fontsize=18, y=1.03)
        fig.colorbar(sm, ax=None, orientation='vertical', shrink=0.75)
        plt.tight_layout()
        plt.savefig('pics/' + title + '.png')
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

        fig3, ax = plt.subplots(figsize=(7, 7), dpi=600)
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

        fig1, axs = plt.subplots(ncols=2, nrows=2, figsize=(7, 7), dpi=600)
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


    def plot_louvain(self, g, plot_network, title, show_labels=True):
        t = g
        commun_louvain = community_louvain.best_partition(t)
        communities = set(dict(commun_louvain).values())
        number_of_communities = len(communities)
        pos = nx.kamada_kawai_layout(t)
        cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName", ["navy", "green", "yellow", "red"])
        cmap = cm.get_cmap('tab10', max(commun_louvain.values()) + 1)
        sm = plt.cm.ScalarMappable(cmap=cm1)
        color_values = np.array(list(commun_louvain.values())) / max(commun_louvain.values())
        fig, ax = plt.subplots(figsize=(8, 8), dpi=600)
        ax = self.loseAxes(ax, 2)
        nx.draw_networkx_nodes(nx.MultiDiGraph(t), plot_network, commun_louvain.keys(), node_size=200,
                               cmap=plt.get_cmap('jet'), node_color=list(color_values), alpha=0.5)
        nx.draw_networkx_edges(nx.MultiDiGraph(t), plot_network, alpha=0.05, width=0.5, connectionstyle="arc3,rad=0.25", arrowstyle='-')
        label_options = {"ec": "k", "fc": "white", "alpha": 0.25}
        if show_labels:
            labels = {}
            for node in t.nodes():
                labels[node] = node
            nx.draw_networkx_labels(t, pos, labels, font_size=9, font_color='k')
        plt.title(title)
        fig.colorbar(sm, ax=None, orientation='vertical', shrink=0.75)
        plt.tight_layout()
        plt.savefig('pics/' + title + '.png')
        plt.show()
        print('Number of clusters is: ', number_of_communities)


    def plot_graph_chars(self, g):
        degree = list(dict(g.degree()).values())
        degree_dist = {}
        for key in degree:
            if key in degree_dist.keys():
                degree_dist[key] += 1
            else:
                degree_dist[key] = 1

        fig = plt.figure()
        plt.bar(degree_dist.keys(), degree_dist.values())
        plt.ylabel('Distribution (# Occurrences)')
        plt.xlabel('Degree')
        plt.title('Degree Distribution')
        plt.show()

        commun_louvain = community.best_partition(g)
        nodes = list(g.nodes())
        edges = list(g.edges())
        communities = set(dict(commun_louvain).values())
        number_of_communities = len(communities)
        communities_nodes = {}

        # create dictionaries with community IDs as keys
        # and members nodes as values
        for node in nodes:
            if commun_louvain[node] in communities_nodes.keys():
                communities_nodes[commun_louvain[node]].append(node)
            else:
                communities_nodes[commun_louvain[node]] = [node]

        connection_density = np.zeros(shape=(number_of_communities, number_of_communities))
        for i in communities_nodes.keys():
            for j in communities_nodes.keys():
                if i != j:
                    edges = list(nx.edge_boundary(g, communities_nodes[i], communities_nodes[j]))
                    connection_density[i, j] = len(edges)
                else:
                    edges = list(nx.edge_boundary(g, communities_nodes[i], communities_nodes[i]))
                    connection_density[i, j] = len(edges)

        for i in communities_nodes.keys():
            for j in communities_nodes.keys():
                if i != j:
                    denom = len(communities_nodes[i]) * len(communities_nodes[j])
                else:
                    denom = (len(communities_nodes[i]) * (len(communities_nodes[i]) - 1)) / 2
                connection_density[i, j] /= denom

        fig = plt.figure()
        plt.imshow(connection_density, cmap='bwr', interpolation='nearest')
        plt.title("Community Connection Density Heatmap")
        plt.xlabel('Community ID')
        plt.ylabel('Community ID')
        plt.xticks(range(number_of_communities))
        plt.yticks(range(number_of_communities))
        plt.show()


    def plot_graph_groups_heatmap(self, g, groups_dict, title):
        nodes = list(g.nodes())
        edges = list(g.edges())
        communities = set(groups_dict.keys())
        number_of_communities = len(communities)
        communities_nodes = groups_dict.copy()
        groups_to_ints = {'Privacy': 0, 'PoW': 1, 'PoS': 2, 'NFT': 3, 'DEX': 4,
                         'DeFi': 5}

        cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName", ["navy", "green", "yellow", "red"])
        sm = plt.cm.ScalarMappable(cmap=cm1)

        connection_density = np.zeros(shape=(number_of_communities, number_of_communities))
        for i in communities_nodes.keys():
            for j in communities_nodes.keys():
                if i != j:
                    edges = list(nx.edge_boundary(g, communities_nodes[i], communities_nodes[j]))
                    connection_density[groups_to_ints[i], groups_to_ints[j]] = len(edges)
                else:
                    edges = list(nx.edge_boundary(g, communities_nodes[i], communities_nodes[i]))
                    connection_density[groups_to_ints[i], groups_to_ints[j]] = len(edges)

        for i in communities_nodes.keys():
            for j in communities_nodes.keys():
                if i != j:
                    denom = len(communities_nodes[i]) * len(communities_nodes[j])
                else:
                    denom = (len(communities_nodes[i]) * (len(communities_nodes[i]) - 1)) / 2
                connection_density[groups_to_ints[i], groups_to_ints[j]] /= denom

        fig, ax = plt.subplots(figsize=(7, 7), dpi=600)
        plt.imshow(connection_density, cmap=cm1, interpolation='nearest')
        plt.title(title, fontsize=14)
        fig.colorbar(sm, ax=None, orientation='vertical', shrink=0.70)
        ax.set_xticks(range(number_of_communities), fontsize=14)
        ax.set_yticks(range(number_of_communities), fontsize=14)
        ax.set_xticklabels(list(groups_dict.keys()), rotation=45, fontsize=14)
        ax.set_yticklabels(list(groups_dict.keys()), rotation=45, fontsize=14)
        plt.tight_layout()
        plt.savefig('pics/' + title + '.png')
        plt.show()

def run():
    data_pro = DataProvider()
    graph_pro = GraphProvider()
    plot_pro = PlotProvider()
    start_date = date(2020, 1, 1)
    end_date = date(2021, 12, 31)
    increment = 1
    window = 15
    dates = data_pro.get_dates(start_date, end_date, increment)
    cryptos = data_pro.get_data(start_date, end_date)
    interes_list = {'btc': 'b', 'eth': 'r', 'ada': 'g', 'xrp': 'orange'}
    events = {date(2021, 9, 24): 'China cracks down further on Bitcoin by banning mining.',
              date(2021, 7, 29): 'Germany permits financial institutions to hold up to 20% of their assets in crypto.',
              date(2021, 5, 12): 'Tesla U-turns on its decision to accept Bitcoin as payment.',
              date(2021, 2, 8): 'Tesla reveals it has bought $1.5 billion of Bitcoin.',
              date(2020, 9, 29): 'Jack Dorsey publicly supports Bitcoin.',
              date(2020, 3, 12): 'COVID-19 affects US markets.'}
    bull_runs = [[date(2019, 12, 18), date(2020, 2, 13)],
                 [date(2020, 3, 17), date(2021, 4, 14)],
                 [date(2021, 7, 21), date(2021, 11, 9)]]
    crypto_groups = {'Privacy': ['xmr', 'zec'],
                     'PoW': ['bch', 'bsv', 'btc', 'dash', 'doge', 'etc', 'eth', 'ltc', 'miota', 'wbtc', 'xmr', 'zec'],
                     'PoS': ['ada', 'algo', 'atom', 'eos', 'eth', 'ht', 'nexo', 'waves', 'xtz'],
                     'NFT': ['chz', 'enj', 'mana', 'omi', 'theta', 'xtz'],
                     'DEX': ['lrc', 'rune', 'xlm'],
                     'DeFi': ['bat', 'cel', 'cusdc', 'dai', 'ftm', 'hbar', 'ht', 'link', 'lrc', 'luna', 'matic', 'mkr',
                              'rune', 'stx', 'wbtc']}
    #'ETH-based': ['bat', 'cdai', 'cel', 'cusdc', 'dai', 'etc', 'eth', 'kcs', 'link', 'lrc', 'nexo',
    #'okb', 'one', 'qnt', 'tusd', 'usdc', 'usdp', 'wbtc', 'xtz'],
    #'BTC-based': ['bch', 'bsv', 'btc', 'doge', 'stx']}
    #'Stable coins': ['busd', 'dai', 'tusd', 'usdc', 'usdp', 'usdt', 'cdai'],
    #[date(2015, 1, 14), date(2017, 6, 11)],
    #[date(2017, 7, 16), date(2017, 9, 1)],
    #[date(2017, 9, 14), date(2017, 12, 16)],
    #[date(2018, 2, 6), date(2018, 3, 6)],
    #[date(2018, 4, 7), date(2018, 5, 6)],
    #[date(2018, 6, 29), date(2018, 7, 25)],
    #[date(2018, 12, 15), date(2019, 6, 27)
    #date(2018, 5, 5): 'Warren Buffet warns investors to steer clear of Bitcoin'.
    #date(2018, 5, 2): 'The Chinese government states its intent to ban foreign cryptocurrency exchange platforms in addition to domestic exchanges.'
    #date(2017, 11, 29): 'John McAfee Tweets a prediction that Bitcoin will reach $1 million by 2020.'
    #date(2015, 10, 22): 'The EU declares Bitcoin is to be treated as a currency in Europe.'
    graphs = graph_pro.calc_graphs(dates, cryptos, data_pro, graph_pro, window)

    degree_centrality = []
    degree_centrality_market = []
    for g in graphs:
        degree_dict = nx.degree_centrality(g.get_graph())
        degree_centrality.append(degree_dict)
        degree_centrality_market.append(np.mean(list(degree_dict.values())))

    plot_pro.plot_metric(dates, degree_centrality, degree_centrality_market,
                         xlable="Date", ylable="Degree centrality", title="Degree centrality", window=window,
                         interes_list=interes_list, major_events={})
    plot_pro.plot_metric(dates, degree_centrality, degree_centrality_market,
                         xlable="Date", ylable="Degree centrality", title="Degree centrality", window=window,
                         interes_list={}, major_events=events)

    betweenness_centrality = []
    betweenness_centrality_market = []
    for g in graphs:
        degree_dict = nx.betweenness_centrality(g.get_graph())
        betweenness_centrality.append(degree_dict)
        betweenness_centrality_market.append(np.mean(list(degree_dict.values())))

    betweenness_centrality_market = [sum(d.values()) / len(d) for d in betweenness_centrality]
    plot_pro.plot_metric(dates, betweenness_centrality, betweenness_centrality_market,
                         xlable="Date", ylable="Betweenness centrality", title="Betweenness centrality", window=window,
                         interes_list=interes_list, major_events={})
    plot_pro.plot_metric(dates, betweenness_centrality, betweenness_centrality_market,
                         xlable="Date", ylable="Betweenness centrality", title="Betweenness centrality", window=window,
                         interes_list={}, major_events=events)

    clustering_coeff = []
    clustering_coeff_market = []
    for g in graphs:
        coeff_dict = nx.clustering(g.get_graph())
        clustering_coeff.append(coeff_dict)
        clustering_coeff_market.append(np.mean(list(coeff_dict.values())))

    plot_pro.plot_metric(dates, clustering_coeff, clustering_coeff_market,
                         xlable="Date", ylable="Clustering coefficient", title="Clustering coefficient", window=window, major_events=events)

    shortest_path = []
    shortest_path_market = []
    for g in graphs:
        largest_cc = max(nx.connected_components(g.get_graph()), key=len)
        largest_cc_graph = g.get_graph().subgraph(largest_cc)
        shortest_dict = dict(nx.shortest_path_length(largest_cc_graph))
        for k in shortest_dict:
            shortest_dict[k] = np.mean(list(shortest_dict[k].values()))
        shortest_path.append(shortest_dict)
        shortest_path_market.append(np.mean(list(shortest_dict.values())))

    plot_pro.plot_metric(dates, shortest_path, shortest_path_market,
                         xlable="Date", ylable="Avg. shortest path", title="Avg. shortest path in largest connected component", window=window, major_events=events)

    plot_pro.plot_metric_bull(dates, degree_centrality_market, xlable="Date", ylable="Degree centrality",
                             title="Degree centrality and market cycles", window=window, cycles=bull_runs)
    plot_pro.plot_metric_bull(dates, betweenness_centrality_market, xlable="Date", ylable="Betweenness centrality",
                              title="Betweenness centrality and market cycles", window=window,
                              cycles=bull_runs)

    nodes_pos = None

    for i, d in enumerate([date(2021, 4, 1), date(2020, 3, 20)]):
        for j in range(len(graphs)):
            start, end = graphs[j].get_dates()
            if start == d:
                if nodes_pos == None:
                    nodes_pos = nx.kamada_kawai_layout(graphs[j].get_graph())
                    plot_pro.plot_network(nx.degree_centrality, graphs[j].get_graph(), nodes_pos,
                                          title='Cryptocurrencies network topology\nfrom ' + start.strftime('%-d %b %y') + ' to ' + end.strftime('%-d %b %y'), show_labels=True)
                plot_pro.plot_network(nx.degree_centrality, graphs[j].get_graph(), nodes_pos, title='Degree centrality from ' + start.strftime('%-d %b %y') + ' to ' + end.strftime('%-d %b %y'))
                plot_pro.plot_network(nx.betweenness_centrality, graphs[j].get_graph(), nodes_pos, title='Betweenness centrality from ' + start.strftime('%-d %b %y') + ' to ' + end.strftime('%-d %b %y'))
                #plot_pro.plot_graph_chars(graphs[j].get_graph())
                #plot_pro.plot_degree_corr(graphs[j].get_graph())
                #plot_pro.plot_degree_dst(graphs[j].get_graph())
                plot_pro.plot_graph_groups_heatmap(graphs[j].get_graph(), crypto_groups, 'Groups connections density heatmap from ' + start.strftime('%-d %b %y') + ' to ' + end.strftime('%-d %b %y'))
                plot_pro.plot_louvain(graphs[j].get_graph(), nodes_pos, "Community Detection using Louvain method")

run()