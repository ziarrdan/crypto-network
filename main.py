from datetime import date
import DataProvider
import GraphProvider
import Plotter
import os
import pickle

GRAPHS_FILE = 'data-post/graphs.pkl'
SELECT_CRYPTOS = {'btc': 'b', 'eth': 'r', 'ada': 'g', 'xrp': 'orange'}
EVENTS = {date(2023, 6, 21): 'Major firms refile application for Spot Bitcoin ETF.',
          date(2022, 11, 2): 'Coindesk publishes an article raising concerns about the financial health of FTX.',
          date(2021, 9, 24): 'China cracks down further on Bitcoin by banning mining.',
          date(2021, 2, 8): 'Tesla reveals it has bought $1.5 billion of Bitcoin.',
          date(2020, 3, 12): 'COVID-19 affects US markets.'}

# Our source for major crypto events
# https://thedefiant.io/charting-history-s-rhyme-understanding-the-cyclicality-of-crypto-markets
BULL_RUNS = [[date(2020, 3, 16), date(2021, 11, 10)], [date(2022, 11, 21), date(2024, 1, 1)]]

# Our source for crypto categories: https://coinmarketcap.com/cryptocurrency-category/
CRYPTO_GROUPS = {'Store of Value': ['btc', 'bch', 'bsv', 'mkr', 'lunc', 'dcr'],
                 'Payments': ['bnb', 'doge', 'trx', 'bch', 'leo', 'hbar', 'cro', 'mana', 'lunc', 'nexo', 'zil', 'gas',
                              'ht', 'btg', 'xem', 'tfuel', 'xrp', 'ltc', 'xlm', 'xmr', 'bsv', 'eos', 'zec', 'dash',
                              'dcr', 'wbtc', ''],
                 'Smart Contract': ['eth', 'bnb', 'ada', 'avax', 'link', 'icp', 'xlm', 'inj', 'etc', 'vet', 'stx',
                                    'algo', 'zil', 'waves', 'neo', 'eos', 'xtz', 'qtum', 'ftm', 'xdc'],
                 'CEX': ['bnb', 'okb', 'cro', 'kcs', 'gt', 'ht', 'mx'],
                 'DEX': ['uni', 'xlm', 'inj', 'rune', 'snx', 'zrx'],
                 'NFT': ['icp', 'imx', 'stx', 'rndr', 'flow', 'theta', 'mana', 'chz', 'fet', 'enj'],
                 'DeFi': ['avax', 'matic', 'link', 'icp', 'dai', 'uni', 'inj', 'ldo', 'stx', 'grt', 'rune', 'aave',
                          'mkr', 'ftm', 'theta', 'ocean', 'snx', 'kava', 'rpl', 'trb', 'lrc', 'bat', 'band', 'ankr'
                          'wbtc'],
                 'Privacy': ['xmr', 'icp']}

CUTOFF = 0.5


def save_file(graphs, filename):
    with open(filename, 'wb') as f:
        pickle.dump(graphs, f)


def load_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def run():
    start_date = date(2020, 1, 1)
    end_date = date(2024, 1, 1)
    increment = 1
    window = 15

    data_provider = DataProvider.DataProvider(start_date, end_date)
    dates = data_provider.get_dates(increment)
    cryptos = data_provider.get_data()

    if os.path.isfile(GRAPHS_FILE):
        graphs = load_file(GRAPHS_FILE)
    else:
        graph_provider = GraphProvider.GraphProvider(dates, cryptos, data_provider, window)
        graphs = graph_provider.calc_graphs(CUTOFF)
        save_file(graphs, GRAPHS_FILE)

    plot_provider = Plotter.Plotter()
    plot_provider.plot_basic_metrics(graphs, dates, window, BULL_RUNS, EVENTS, SELECT_CRYPTOS)
    plot_provider.plot_composite_metrics(graphs, CRYPTO_GROUPS)


run()