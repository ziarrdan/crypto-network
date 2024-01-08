import community as community_louvain
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcol
import matplotlib.cm as cm
from datetime import date
from scipy import stats
import collections
import community
import networkx as nx
import numpy as np


class Plotter:
    def lose_axes(self, ax, group=1):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if group == 2:
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        else:
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

        return ax

    def plot_metric(self, fig_comp, ax_comp, dates, degree_metric, degree_metric_market, xlable, ylable, title, window,
                    interest_list={}, major_events={}):
        fig, ax = plt.subplots(figsize=(8, 4), dpi=600)
        for coin, coin_color in interest_list.items():
            plt.plot(dates[window:], [d.get(coin) for d in degree_metric], color=coin_color, linewidth=.5,
                     linestyle='--',
                     label=coin)
        plt.plot(dates[window:], degree_metric_market, color='k', linewidth=1.25, label='Market')
        for key in major_events.keys():
            plt.axvline(x=key, color='k', linestyle='--', linewidth=1)
        if len(interest_list) == 0:
            ax.set_xticks(list(major_events.keys()))
        myFmt = mdates.DateFormatter('%b %y')
        ax = self.lose_axes(ax, 1)
        ax.xaxis.set_major_formatter(myFmt)
        plt.yticks(fontsize=12)
        plt.xticks(rotation=25, fontsize=12)
        plt.legend(ncol=len(interest_list) + 1)
        plt.xlabel(xlable, fontsize=12)
        plt.ylabel(ylable, fontsize=12)
        plt.title(title, fontsize=16, y=1.03)
        plt.close()

        for coin, coin_color in interest_list.items():
            ax_comp.plot(dates[window:], [d.get(coin) for d in degree_metric], color=coin_color, linewidth=.5,
                         linestyle='--',
                         label=coin)
        ax_comp.plot(dates[window:], degree_metric_market, color='k', linewidth=1.25, label='Market')
        for key in major_events.keys():
            ax_comp.axvline(x=key, color='k', linestyle='--', linewidth=1)
        if len(interest_list) == 0:
            ax_comp.set_xticks(list(major_events.keys()), rotation=25, fontsize=22)
            ax_comp.xaxis.set_tick_params(labelsize=22)
            ax_comp.yaxis.set_tick_params(labelsize=22)
        myFmt = mdates.DateFormatter('%b %y')
        ax_comp = self.lose_axes(ax_comp, 1)
        ax_comp.xaxis.set_major_formatter(myFmt)
        ax_comp.set_title(title, fontsize=24, y=1.03)

    def plot_network_with_labels(self, metric_func, graph, plot_network, title, edge_widths=0.5):
        fig, ax = plt.subplots(figsize=(9, 7), dpi=600)
        degree_metric = metric_func(graph)
        ax = self.lose_axes(ax, 2)
        graph = graph.subgraph(['btc', 'eth', 'bnb', 'xrp', 'ada', 'doge', 'trx', 'matic', 'link', 'wbtc',
                                'ltc', 'bch', 'atom', 'xlm', 'okb', 'hbar', 'etc', 'xmr', 'cro',
                                'vet', 'stx', 'rune', 'bsv', 'mkr', 'algo', 'qnt', 'ftm', 'theta', 'snx',
                                'kcs', 'xtz', 'kava', 'mana', 'neo', 'klay', 'eos', 'iota', 'xdc', 'lunc'])
        edges = graph.edges()
        colors = [degree_metric[node] / max(degree_metric.values()) for node in graph.nodes()]
        sizes = [500 for _ in colors]
        cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName", ["navy", "green", "yellow", "red"])
        nodes = nx.draw_networkx_nodes(nx.MultiDiGraph(graph),
                                       pos=plot_network,
                                       node_color=colors,
                                       node_size=sizes,
                                       cmap=cm1)
        nx.draw_networkx_edges(nx.MultiDiGraph(graph),
                               pos=plot_network,
                               edgelist=edges,
                               edge_color='k',
                               width=edge_widths,
                               connectionstyle="arc3,rad=0.25",
                               arrowstyle='-',
                               alpha=0.25)
        labels = {}
        for i, node in enumerate(graph.nodes()):
            labels[node] = node
        nx.draw_networkx_labels(graph, plot_network, labels, font_size=8, font_color='white')
        plt.title(title, fontsize=14)
        cbar = plt.colorbar(nodes, shrink=0.6)
        cbar.ax.tick_params(labelsize=12)
        plt.savefig('pics/' + title + '.png', bbox_inches='tight')
        plt.close()

    def plot_network(self, fig_comp, ax_comp, metric_func, graph, plot_network, title, edge_widths=0.5):
        fig, ax = plt.subplots(figsize=(8, 7), dpi=600)
        degree_metric = metric_func(graph)
        ax = self.lose_axes(ax, 2)
        edges = graph.edges()
        colors = [degree_metric[node] / max(degree_metric.values()) for node in graph.nodes()]
        sizes = [200 * i if i > 0 else 10 for i in colors]
        cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName", ["navy", "green", "yellow", "red"])
        sm = plt.cm.ScalarMappable(cmap=cm1)
        node_colors = cm1(colors)
        nodes = nx.draw_networkx_nodes(nx.MultiDiGraph(graph), pos=plot_network, node_color=colors, node_size=sizes,
                                       cmap=cm1)
        nx.draw_networkx_edges(nx.MultiDiGraph(graph), pos=plot_network, edgelist=edges, edge_color='k',
                               width=edge_widths, connectionstyle="arc3,rad=0.25", arrowstyle='-', alpha=0.25)
        plt.title(title, fontsize=16)
        cbar = plt.colorbar(nodes, shrink=0.6)
        cbar.ax.tick_params(labelsize=12)
        plt.close()

        if fig_comp != None and ax_comp != None:
            ax_comp.set_title(title, fontsize=24, y=1.03, va='center')
            ax_comp = self.lose_axes(ax_comp, 2)
            ax_comp.set_aspect('equal')
            nx.draw_networkx_nodes(nx.MultiDiGraph(graph), pos=plot_network, node_color=node_colors, node_size=sizes,
                                   ax=ax_comp)
            nx.draw_networkx_edges(nx.MultiDiGraph(graph), pos=plot_network, edgelist=edges, edge_color='k',
                                   width=edge_widths, connectionstyle="arc3,rad=0.25", arrowstyle='-', alpha=0.25,
                                   ax=ax_comp)

        return sm

    def plot_graph_groups_heatmap(self, fig_comp, ax_comp, g, groups_dict, groups_to_ints, title):
        communities = set(groups_dict.keys())
        number_of_communities = len(communities)
        communities_nodes = groups_dict.copy()

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

        fig, ax = plt.subplots(figsize=(8, 7), dpi=600)
        plt.imshow(connection_density, cmap=cm1, interpolation='nearest')
        plt.title(title, fontsize=16)
        fig.colorbar(sm, ax=None, orientation='vertical', shrink=0.6)
        ax.set_xticks(range(number_of_communities), fontsize=22)
        ax.set_yticks(range(number_of_communities), fontsize=22)
        ax.set_xticklabels(list(groups_dict.keys()), rotation=45, fontsize=22)
        ax.set_yticklabels(list(groups_dict.keys()), rotation=45, fontsize=22)
        plt.close()

        ax_comp.set_title(title, fontsize=24, y=1.03, va='center')
        ax_comp.set_aspect('equal')
        ax_comp.imshow(connection_density, cmap=cm1, interpolation='nearest')

        return sm

    def plot_basic_metrics(self, graphs, dates, window, bull_runs, events, interest_list):
        fig, axs = plt.subplots(2, 2, figsize=(20, 10), dpi=600, layout='constrained')
        degree_centrality = []
        degree_centrality_market = []
        for g in graphs:
            degree_dict = nx.degree_centrality(g.get_graph())
            degree_centrality.append(degree_dict)
            degree_centrality_market.append(np.mean(list(degree_dict.values())))
        self.plot_metric(fig, axs[0][0], dates, degree_centrality, degree_centrality_market,
                         xlable="Date", ylable="Degree centrality", title="Degree centrality", window=window,
                         interest_list={}, major_events=events)

        axs[0][0].xaxis.set_tick_params(labelbottom=False)

        betweenness_centrality = []
        betweenness_centrality_market = []
        for g in graphs:
            degree_dict = nx.betweenness_centrality(g.get_graph(), normalized=True)
            betweenness_centrality.append(degree_dict)
            betweenness_centrality_market.append(np.mean(list(degree_dict.values())))
        betweenness_centrality_market = [sum(d.values()) / len(d) for d in betweenness_centrality]
        self.plot_metric(fig, axs[0][1], dates, betweenness_centrality, betweenness_centrality_market,
                         xlable="Date", ylable="Betweenness centrality", title="Betweenness centrality",
                         window=window,
                         interest_list={}, major_events=events)

        axs[0][1].xaxis.set_tick_params(labelbottom=False)

        clustering_coeff = []
        clustering_coeff_market = []
        for g in graphs:
            coeff_dict = nx.clustering(g.get_graph())
            clustering_coeff.append(coeff_dict)
            clustering_coeff_market.append(np.mean(list(coeff_dict.values())))
        self.plot_metric(fig, axs[1][0], dates, clustering_coeff, clustering_coeff_market,
                         xlable="Date", ylable="Clustering coefficient", title="Clustering coefficient",
                         window=window,
                         major_events=events)

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
        self.plot_metric(fig, axs[1][1], dates, shortest_path, shortest_path_market,
                         xlable="Date", ylable="Avg. shortest path",
                         title="Avg. shortest path in largest connected component", window=window,
                         major_events=events)

        fig.supxlabel('Time', fontsize=26)
        fig.suptitle("Centrality metrics", fontsize=26)
        fig.savefig("pics/composite" + str(3) + ".png", bbox_inches='tight')
        plt.close(fig)

    def plot_composite_metrics(self, graphs, crypto_groups):
        groups_to_ints = {'Store of Value': 0, 'Payments': 1, 'Smart Contract': 2, 'CEX': 3, 'DEX': 4,
                          'NFT': 5, 'DeFi': 6, 'Privacy': 7}
        nodes_pos = None
        plot_option = 0
        for fig, axs in [(plt.subplots(2, 2, figsize=(22, 20), dpi=600, layout='constrained')),
                         (plt.subplots(2, 2, figsize=(22, 20), dpi=600, layout='constrained')),
                         (plt.subplots(2, 2, figsize=(22, 20), dpi=600, layout='constrained'))]:
            for i, d in enumerate([date(2021, 3, 20), date(2020, 9, 20), date(2020, 6, 20), date(2020, 3, 20)]):
                for j in range(len(graphs)):
                    start, end = graphs[j].get_dates()
                    if start == d:
                        if nodes_pos is None:
                            nodes_pos = nx.kamada_kawai_layout(graphs[j].get_graph())

                        ax = axs[1 - (i // 2)][1 - (i % 2)]
                        if plot_option == 0:
                            sm = self.plot_network(fig, ax, nx.degree_centrality, graphs[j].get_graph(),
                                                   nodes_pos, title=start.strftime(
                                    '%B %-d, %Y') + ' to ' + end.strftime('%B %-d, %Y'))
                        elif plot_option == 1:
                            sm = self.plot_network(fig, ax, nx.betweenness_centrality,
                                                   graphs[j].get_graph(), nodes_pos,
                                                   title=start.strftime(
                                                       '%B %-d, %Y') + ' to ' + end.strftime('%B %-d, %Y'))
                        elif plot_option == 2:
                            sm = self.plot_graph_groups_heatmap(fig, ax, graphs[j].get_graph(), crypto_groups,
                                                                groups_to_ints,
                                                                start.strftime(
                                                                    '%B %-d, %Y') + ' to ' + end.strftime(
                                                                    '%B %-d, %Y'))
                            if i == 1:
                                ax.set_yticks(range(len(groups_to_ints)), fontsize=22)
                                ax.set_yticklabels(list(groups_to_ints.keys()), rotation=45, fontsize=22)
                                ax.set_xticks(range(len(groups_to_ints)), fontsize=22)
                                ax.set_xticklabels(list(groups_to_ints.keys()), rotation=45, fontsize=22)
                            elif i == 0:
                                ax.yaxis.set_tick_params(labelbottom=False)
                                ax.set_xticks(range(len(groups_to_ints)), fontsize=22)
                                ax.set_xticklabels(list(groups_to_ints.keys()), rotation=45, fontsize=22)
                            elif i == 3:
                                ax.xaxis.set_tick_params(labelleft=False)
                                ax.set_yticks(range(len(groups_to_ints)), fontsize=22)
                                ax.set_yticklabels(list(groups_to_ints.keys()), rotation=45, fontsize=22)
                            else:
                                ax.xaxis.set_tick_params(labelbottom=False)
                                ax.yaxis.set_tick_params(labelleft=False)

                        if start == date(2021, 3, 20):
                            self.plot_network_with_labels(nx.degree_centrality, graphs[j].get_graph(),
                                                          nodes_pos,
                                                          title='Cryptocurrency network topology\nfrom ' + start.strftime(
                                                              '%B %-d, %Y') + ' to ' + end.strftime(
                                                              '%B %-d, %Y') + ' \nfor top 40 coins by market cap')

            if plot_option == 0:
                title = "Degree centrality"
            elif plot_option == 1:
                title = "Betweenness centrality"
            elif plot_option == 2:
                title = "Groups connections density heatmap"
            fig.suptitle(title, fontsize=26)
            cbar = fig.colorbar(sm, ax=axs[:, 1], orientation='vertical', shrink=0.3)
            cbar.ax.tick_params(labelsize=22)
            fig.savefig("pics/composite" + str(plot_option) + ".png", bbox_inches='tight')
            plt.close(fig)
            plot_option += 1

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

        fig3, ax = plt.subplots(figsize=(8, 7), dpi=600)
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

        # k_nn_bar for a neutral degree is known to be (k_bar + (std / k_bar)), hence:
        k = list(dict(g.degree()).values())
        k_nn_bar = np.mean(k) + (np.std(k) / np.mean(k))

        dist_for_ttest = np.array(list(avg_neighbor_for_ks.values()))
        stats.ttest_1samp(dist_for_ttest, k_nn_bar)

    def plot_degree_dst(self, g):
        # Plot the out-degree distribution
        degree_sequence = reversed(sorted([d for n, d in g.degree()], reverse=True))  # degree sequence
        degreeCount = collections.Counter(degree_sequence)
        deg, in_cnt = zip(*degreeCount.items())
        cnt = np.array(in_cnt) / g.number_of_nodes()
        degree = [d for n, d in g.degree() if d != 0]

        fig1, axs = plt.subplots(ncols=2, nrows=2, figsize=(8, 7), dpi=600)
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
        bins = [2 ** n for n in range(n)]
        N = {}
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
        ax = self.lose_axes(ax, 2)
        nx.draw_networkx_nodes(nx.MultiDiGraph(t), plot_network, commun_louvain.keys(), node_size=200,
                               cmap=plt.get_cmap('jet'), node_color=list(color_values), alpha=0.5)
        nx.draw_networkx_edges(nx.MultiDiGraph(t), plot_network, alpha=0.05, width=0.5, connectionstyle="arc3,rad=0.25",
                               arrowstyle='-')
        label_options = {"ec": "k", "fc": "white", "alpha": 0.25}
        if show_labels:
            labels = {}
            for node in t.nodes():
                labels[node] = node
            nx.draw_networkx_labels(t, pos, labels, font_size=9, font_color='k')
        plt.title(title)
        fig.colorbar(sm, ax=None, orientation='vertical', shrink=0.6)
        plt.tight_layout()
        plt.savefig('pics/' + title + '.png')
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

    def plot_metric_bull_bear(self, dates, degree_metric, xlable, ylable, title, window, cycles):
        fig, ax = plt.subplots(figsize=(8, 4), dpi=600)
        plt.plot(dates[window:], np.array(degree_metric), color='k', linewidth=1.25)
        for start, end in cycles:
            plt.axvline(x=start, color='g', linestyle='--', linewidth=1)
            plt.axvline(x=end, color='r', linestyle='--', linewidth=1)
        myFmt = mdates.DateFormatter('%b %y')
        ax = self.lose_axes(ax, 1)
        ax.xaxis.set_major_formatter(myFmt)
        plt.yticks(fontsize=12)
        plt.xticks(rotation=25, fontsize=12)
        plt.xlabel(xlable, fontsize=12)
        plt.ylabel(ylable, fontsize=12)
        plt.title(title, fontsize=16, y=1.03)
        plt.tight_layout()
        plt.savefig('pics/' + title + '.png')
        plt.close()
