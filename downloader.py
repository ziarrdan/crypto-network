from pycoingecko import CoinGeckoAPI
import datetime
import pandas as pd
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir('data-master/mine/') if isfile(join('data-master/mine/', f))]
onlyfiles = [f.split('.')[0].strip() for f in onlyfiles]

def download():
    cg = CoinGeckoAPI()
    coin_list = cg.get_coins_markets(vs_currency='usd')
    top_100 = [dic['id'] for dic in coin_list]
    top_100_symbol = [dic['symbol'] for dic in coin_list]
    for c, coin in enumerate(top_100):
        symb = top_100_symbol[c]
        if not symb in onlyfiles:
            coin_market = cg.get_coin_market_chart_by_id(id=coin, vs_currency='usd', days=2000)
            for d in range(len(coin_market['prices'])):
                coin_market['prices'][d][0] = datetime.datetime.fromtimestamp(coin_market['prices'][d][0] / 1000)
            coin_df = pd.DataFrame(coin_market['prices'], columns=['Date', 'Open'])
            coin_df.to_csv('data-master/mine/' + symb + '.csv')


download()